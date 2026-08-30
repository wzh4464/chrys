# Copyright (c) 2026 Chrys. All rights reserved.

"""Machine-enforced hygiene rules for the test suite."""

from __future__ import annotations

import ast
import functools
import importlib.metadata
import re
import tomllib
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from tests.support.ci import CI_LINUX_ONLY
from tests.support.paths import REPO_ROOT, SRC_ROOT, TESTS_ROOT

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY

# Shard count for the sweep test. One monolithic sweep measured 53-55s of
# worker time on contended CI draws — 90% of the global 60s per-test timeout —
# and was the suite's #1 tail pole on every platform. Shards are disjoint by
# construction (stable path hash modulo), so every file is checked exactly once
# and worksteal can spread the shards across workers.
_SWEEP_SHARDS = 4

_POLLING_HELPER_NAMES = {"_wait_until", "_wait_for", "_poll_until", "_wait_for_ui", "_eventually"}
_LOCAL_POLLING_ALLOWLIST = {
    (Path("tests/orchestration/sub_agents/test_controller.py"), "_wait_until"),
    (Path("tests/integration/engine/test_engine_integration.py"), "_wait_until"),
}
_OPTIONAL_IMPORT_ALLOWLIST = {
    Path("src/chrys/foundation/observability/exporters.py"): (
        "every symbol in the module subclasses an OTel SDK type, so it cannot be written without the extra; "
        "it stays safe because nothing imports it at module scope — observability/setup.py reaches it from "
        "inside a function, under the same gate that decides whether telemetry runs at all"
    ),
}
_ENGINE_START_PATH_ALLOWLIST = {
    Path("tests/support/pipeline_helpers.py"),
}
_RAW_TOOL_LOOP_LAYER_ALLOWLIST = {
    # Constructor/delegation pins that never obtain a final response from the
    # layer, so there is no transcript for the oracle to check.
    (Path("tests/kernel/test_loop.py"), "test_getattr_two_hop_delegation"),
    (Path("tests/kernel/test_loop.py"), "test_getattr_attribute_error_passthrough"),
    (Path("tests/kernel/test_loop.py"), "test_ctor_rejects_chat_middleware"),
    (Path("tests/kernel/test_loop.py"), "test_ctor_defaults_mirror_framework_values"),
    # The wire dies mid-stream, no loop iteration ever lands, and the fallback
    # re-merge is not a loop-landed transcript — the invariant oracle's
    # precondition does not hold for that final response.
    (Path("tests/kernel/test_loop.py"), "test_degenerate_stream_falls_back_to_from_updates"),
    # The checked layer itself wraps the raw layer by deriving from it.
    (Path("tests/support/transcript_invariants.py"), "InvariantCheckedToolLoopLayer"),
}
_DIRECT_TRAJECTORY_PREFIX_ORACLE_ALLOWLIST = {
    # Reader unit tests deliberately compare decoded events with physical
    # slots to pin the distinction the shared oracle must hide from callers.
    Path("tests/foundation/trajectory/test_reader.py"),
    Path("tests/support/trajectory_invariants.py"),
}
_ENTRYPOINT_BOOTSTRAP_EXEMPT = {
    Path("src/chrys/app/cli/serve.py"): "spawns the TUI child, which performs bootstrap_runtime itself",
    Path("src/chrys/app/installer.py"): (
        "copies the PyApp binary and edits PATH without touching sessions, config, or the agent runtime; "
        "bootstrapping would be dead weight and would let a broken config block the installer"
    ),
}

_SUBPROCESS_FUNCTIONS = {
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
}
_DEPENDENCY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_LOWER_TIER_ROOTS = tuple(Path("src/chrys") / tier for tier in ("foundation", "kernel", "service", "orchestration"))

# An explicit decorator registry ties each allowlist to an executable pin.
# Source-text matching is shorter, but a comment or dead reference could satisfy
# it; the registry adds one declaration per pin in exchange for structural proof.
_ALLOWLIST_PIN_REGISTRY: dict[str, list[Callable[..., None]]] = {}


def _pins_allowlist(name: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    def _register(pin: Callable[..., None]) -> Callable[..., None]:
        _ALLOWLIST_PIN_REGISTRY.setdefault(name, []).append(pin)
        return pin

    return _register


def _shard_of(relative_path: Path) -> int:
    """Stable shard assignment: seed-independent, identical on every platform."""
    return zlib.crc32(relative_path.as_posix().encode("utf-8")) % _SWEEP_SHARDS


def _test_sources(shard: int | None = None) -> dict[Path, str]:
    relative = ((path, path.relative_to(REPO_ROOT)) for path in sorted(TESTS_ROOT.rglob("*.py")))
    return {rel: path.read_text(encoding="utf-8") for path, rel in relative if shard is None or _shard_of(rel) == shard}


def _src_sources(shard: int | None = None) -> dict[Path, str]:
    relative = ((path, path.relative_to(REPO_ROOT)) for path in sorted((SRC_ROOT / "chrys").rglob("*.py")))
    return {rel: path.read_text(encoding="utf-8") for path, rel in relative if shard is None or _shard_of(rel) == shard}


@functools.cache
def _global_definition_lines() -> dict[str, int]:
    """Module-level definition lines used in self-contained meta-guard errors."""
    module_path = Path(__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=module_path.as_posix())
    lines: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lines[node.name] = node.lineno
            continue
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for target in targets:
            if isinstance(target, ast.Name):
                lines[target.id] = node.lineno
    return lines


def _definition_location(name: str) -> str:
    relative_module = Path(__file__).relative_to(REPO_ROOT)
    definition_lines = _global_definition_lines()
    line = definition_lines.get(name, definition_lines["_ALLOWLIST_PIN_REGISTRY"])
    return f"{relative_module}:{line}"


def _meta_guard_problem(name: str, problem: str, fix: str) -> str:
    return (
        f"{_definition_location(name)}: {problem}; violates AGENTS.md:129 "
        '("A guard that can\'t go red is worse than none — it is believed."). '
        f"Fix: {fix}"
    )


def _allowlist_target_paths() -> set[Path]:
    paths = set(_ENGINE_START_PATH_ALLOWLIST)
    paths.update(_DIRECT_TRAJECTORY_PREFIX_ORACLE_ALLOWLIST)
    paths.update(_CLASSIFIER_IMPORT_ALLOWLIST)
    paths.update(path for path, _name in _LOCAL_POLLING_ALLOWLIST)
    paths.update(path for path, _scope in _RAW_TOOL_LOOP_LAYER_ALLOWLIST)
    return paths


@functools.cache
def _allowlist_target_trees() -> dict[Path, ast.Module]:
    """Read and parse the five newly pinned allowlists' targets exactly once."""
    trees: dict[Path, ast.Module] = {}
    for relative_path in sorted(_allowlist_target_paths()):
        absolute_path = REPO_ROOT / relative_path
        if absolute_path.is_file():
            trees[relative_path] = ast.parse(
                absolute_path.read_text(encoding="utf-8"),
                filename=relative_path.as_posix(),
            )
    return trees


@functools.cache
def _tree(path: Path, source: str) -> ast.Module:
    return ast.parse(source, filename=path.as_posix())


def _qualified_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _subprocess_call_names(tree: ast.Module) -> set[str]:
    """Resolve supported subprocess functions through ordinary import aliases."""
    names = set(_SUBPROCESS_FUNCTIONS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {"asyncio", "subprocess"}:
                    continue
                bound = alias.asname or alias.name
                names.update(
                    f"{bound}.{qualified.rsplit('.', maxsplit=1)[1]}"
                    for qualified in _SUBPROCESS_FUNCTIONS
                    if qualified.startswith(f"{alias.name}.")
                )
        elif isinstance(node, ast.ImportFrom) and node.module in {"asyncio", "subprocess"}:
            for alias in node.names:
                qualified = f"{node.module}.{alias.name}"
                if qualified in _SUBPROCESS_FUNCTIONS:
                    names.add(alias.asname or alias.name)
    return names


def _statement_stdin_bindings(statement: ast.stmt) -> set[str]:
    """Return the mappings *statement* itself gives a stdin entry."""
    names: set[str] = set()
    if isinstance(statement, ast.Assign):
        if _dict_has_literal_stdin(statement.value):
            names.update(target.id for target in statement.targets if isinstance(target, ast.Name))
        if not _inherits_parent_stdin(statement.value):
            names.update(
                target.value.id
                for target in statement.targets
                if isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and _is_stdin_key(target.slice)
            )
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        target = statement.target
        if isinstance(target, ast.Name) and _dict_has_literal_stdin(statement.value):
            names.add(target.id)
        elif (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and _is_stdin_key(target.slice)
            and not _inherits_parent_stdin(statement.value)
        ):
            names.add(target.value.id)
    elif isinstance(statement, ast.Expr):
        call = statement.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "setdefault"
            and isinstance(call.func.value, ast.Name)
            and call.args
            and _is_stdin_key(call.args[0])
            and not (len(call.args) > 1 and _inherits_parent_stdin(call.args[1]))
        ):
            names.add(call.func.value.id)
    return names


def _unscoped_nodes(statement: ast.stmt) -> Iterator[ast.AST]:
    """Walk *statement* without entering a nested function, class, or lambda."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    stack: list[ast.AST] = [statement]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(
            child
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
        )


def _statement_stdin_invalidations(statement: ast.stmt) -> set[str]:
    """Return mappings whose prior stdin evidence *statement* may destroy.

    Invalidation is the mirror of evidence and takes the opposite quantifier.
    Evidence must *dominate* the call — a default set inside a branch licenses
    nothing, because it may not run. A removal inside a branch is the reverse:
    it may run, and then the child inherits our stdin again. So this walks the
    whole statement rather than the statement alone.

    Only statically certain removals count. Reassigning ``x["stdin"]`` to a
    real handle is not one — otherwise re-defaulting inside a branch would
    read as a retraction of the default above it.
    """
    names: set[str] = set()
    for node in _unscoped_nodes(statement):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
            names.update(
                target.value.id
                for target in targets
                if isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and _is_stdin_key(target.slice)
                and _inherits_parent_stdin(node.value)
            )
        elif isinstance(node, ast.Delete):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
            names.update(
                target.value.id
                for target in node.targets
                if isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and _is_stdin_key(target.slice)
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            if (
                node.func.attr == "clear"
                or (node.func.attr == "pop" and node.args and _is_stdin_key(node.args[0]))
                or (node.func.attr == "update" and node.args and _dict_maps_stdin_to_none(node.args[0]))
            ):
                names.add(node.func.value.id)
    return names


def _dict_maps_stdin_to_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Dict) and any(
        _is_stdin_key(key) and _inherits_parent_stdin(value) for key, value in zip(node.keys, node.values, strict=True)
    )


def _nested_blocks(statement: ast.stmt) -> list[list[ast.stmt]]:
    """Return the statement lists *statement* opens, excluding nested scopes."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return []
    blocks = [
        block for field in ("body", "orelse", "finalbody") if isinstance(block := getattr(statement, field, None), list)
    ]
    if isinstance(statement, ast.Try):
        blocks.extend(handler.body for handler in statement.handlers)
    return blocks


def _stdin_evidence_by_node(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[int, frozenset[str]]:
    """Map every descendant to the mappings already given stdin where it sits.

    A bare parameter is not proof: a wrapper that forwards ``**kwargs`` it
    never defaults leaves its own callers free to omit stdin, and the sweep
    would still be green. Neither is a binding found anywhere in the body — a
    ``setdefault`` *after* the spawn, or one inside a branch that may not run,
    licenses nothing. Evidence therefore has to dominate the call: it must be
    a statement of a block enclosing the call, positioned before it. That is
    weaker than real dominance analysis and stronger than a flat scan, and it
    is what the two live wrappers actually do — one defaults at the top of the
    function, the other builds the mapping before entering the ``try`` that
    spawns.
    """
    evidence: dict[int, frozenset[str]] = {}

    def walk_block(statements: list[ast.stmt], inherited: frozenset[str]) -> None:
        available = inherited
        for statement in statements:
            for child in ast.walk(statement):
                evidence[id(child)] = available
            for block in _nested_blocks(statement):
                walk_block(block, available)
            available = (available - _statement_stdin_invalidations(statement)) | _statement_stdin_bindings(statement)

    walk_block(node.body, frozenset())
    return evidence


def _is_stdin_key(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == "stdin"


def _dict_has_literal_stdin(node: ast.expr) -> bool:
    return isinstance(node, ast.Dict) and any(
        _is_stdin_key(key) and not _inherits_parent_stdin(value)
        for key, value in zip(node.keys, node.values, strict=True)
    )


def _inherits_parent_stdin(node: ast.expr | None) -> bool:
    """Report whether a statically known value leaves the child on our stdin.

    ``stdin=None`` is the inheriting default spelled out, not a decision, so it
    must not count as evidence. Anything the sweep cannot evaluate is left to
    the author.
    """
    return isinstance(node, ast.Constant) and node.value is None


_THREAD_OFFLOAD_FUNCTIONS = {"asyncio.to_thread"}
_PARTIAL_FUNCTIONS = {"functools.partial"}


def _thread_offload_names(tree: ast.Module) -> set[str]:
    """Resolve thread-offload helpers through ordinary import aliases."""
    names = set(_THREAD_OFFLOAD_FUNCTIONS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "asyncio":
                    names.add(f"{alias.asname or alias.name}.to_thread")
        elif isinstance(node, ast.ImportFrom) and node.module == "asyncio":
            names.update(alias.asname or alias.name for alias in node.names if alias.name == "to_thread")
    return names


def _partial_names(tree: ast.Module) -> set[str]:
    """Resolve ``functools.partial`` through ordinary import aliases."""
    names = set(_PARTIAL_FUNCTIONS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "functools":
                    names.add(f"{alias.asname or alias.name}.partial")
        elif isinstance(node, ast.ImportFrom) and node.module == "functools":
            names.update(alias.asname or alias.name for alias in node.names if alias.name == "partial")
    return names


def _spawn_site(
    node: ast.Call,
    call_names: set[str],
    offload_names: set[str],
    partial_names: set[str],
) -> ast.Call | None:
    """Return the call whose keywords reach a subprocess constructor, if any.

    Spawning through ``asyncio.to_thread`` or ``run_in_executor`` hands the
    constructor over as a value, so the visited callee is the offload helper
    and a callee-name test sees nothing. The hook worker in
    ``service/hooks/runner.py`` is spawned exactly that way — the one shape
    this rule most needs to cover.
    """
    if _qualified_name(node.func) in call_names:
        return node
    if _qualified_name(node.func) in offload_names and node.args:
        target = node.args[0]
    elif isinstance(node.func, ast.Attribute) and node.func.attr == "run_in_executor" and len(node.args) > 1:
        target = node.args[1]
    else:
        return None
    if isinstance(target, ast.Call):
        # ``partial(subprocess.Popen, ...)`` carries the keywords itself.
        if _qualified_name(target.func) in partial_names and target.args:
            inner = target.args[0]
            return target if _qualified_name(inner) in call_names else None
        return None
    return node if _qualified_name(target) in call_names else None


class _SubprocessVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        path: Path,
        call_names: set[str],
        offload_names: set[str],
        partial_names: set[str],
        violations: list[str],
    ) -> None:
        self._path = path
        self._call_names = call_names
        self._offload_names = offload_names
        self._partial_names = partial_names
        self._violations = violations
        self._scope_stack: list[dict[int, frozenset[str]]] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._scope_stack.append(_stdin_evidence_by_node(node))
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        spawn = _spawn_site(node, self._call_names, self._offload_names, self._partial_names)
        if spawn is not None and not _subprocess_call_has_stdin(spawn, self._scope_stack):
            self._violations.append(
                f"{self._path}:{node.lineno}: subprocess-explicit-stdin requires stdin= or input= on every "
                "non-interactive subprocess; violates AGENTS.md:39 (every non-interactive subprocess MUST set "
                "stdin explicitly). Fix: pass stdin=subprocess.DEVNULL (or an intentional PIPE/input), or forward "
                "a mapping an enclosing block gives a non-None stdin entry before this call"
            )
        self.generic_visit(node)


def _assert_subprocess_stdin_is_explicit(sources: Mapping[Path, str]) -> None:
    """Keep non-interactive subprocesses detached from ACP's protocol stdin."""
    violations: list[str] = []
    for path, source in sources.items():
        tree = _tree(path, source)
        _SubprocessVisitor(
            path=path,
            call_names=_subprocess_call_names(tree),
            offload_names=_thread_offload_names(tree),
            partial_names=_partial_names(tree),
            violations=violations,
        ).visit(tree)
    assert violations == [], "\n".join(violations)


def _subprocess_call_has_stdin(
    node: ast.Call,
    scope_stack: Sequence[dict[int, frozenset[str]]],
) -> bool:
    if any(
        keyword.arg in {"stdin", "input"} and not _inherits_parent_stdin(keyword.value) for keyword in node.keywords
    ):
        return True
    if not scope_stack:
        return False
    dominating = scope_stack[-1].get(id(node))
    if dominating is None:
        return False
    return any(
        keyword.arg is None and isinstance(keyword.value, ast.Name) and keyword.value.id in dominating
        for keyword in node.keywords
    )


def _normalized_distribution_name(requirement: str) -> str:
    match = _DEPENDENCY_NAME.match(requirement)
    assert match is not None
    return re.sub(r"[-_.]+", "-", match.group()).lower()


# Import roots for the distributions whose root is not just the normalized
# distribution name. This is stated rather than discovered because
# ``packages_distributions()`` only knows what the running interpreter has
# installed, and a distribution that is absent resolves to nothing — its roots
# drop out of the rule silently. That is not hypothetical: pywinpty is
# Windows-marked, so the Linux job that runs this sweep never installs it, and
# a module-scope ``import winpty`` in a lower tier would sail straight through.
# Only public roots are listed; private C-extension and mypyc build artifacts
# (``_yaml``, ``_watchdog_fsevents``, the charset-normalizer hash module) are
# not things source code imports, and their names vary by platform and wheel.
_DISTRIBUTION_IMPORT_ROOTS: Mapping[str, frozenset[str]] = {
    "agent-client-protocol": frozenset({"acp"}),
    # Qualified, because the base API and the optional SDK/exporter share a
    # namespace: subtracting at top-level granularity would cancel the whole
    # of ``opentelemetry`` out of the rule and leave the extra uncovered.
    "opentelemetry-api": frozenset({"opentelemetry"}),
    "opentelemetry-exporter-otlp-proto-grpc": frozenset({"opentelemetry.exporter"}),
    "opentelemetry-instrumentation-logging": frozenset({"opentelemetry.instrumentation"}),
    "opentelemetry-sdk": frozenset({"opentelemetry.sdk"}),
    "pillow": frozenset({"PIL"}),
    "pyjwt": frozenset({"jwt"}),
    "pytest": frozenset({"py", "pytest"}),
    "pytest-xdist": frozenset({"xdist"}),
    "python-docx": frozenset({"docx"}),
    "python-dotenv": frozenset({"dotenv"}),
    "python-pptx": frozenset({"pptx"}),
    "pywinpty": frozenset({"winpty"}),
    "pyyaml": frozenset({"yaml"}),
}


def _import_roots(distribution: str) -> frozenset[str]:
    """Return the import roots a distribution supplies, independent of install state."""
    return _DISTRIBUTION_IMPORT_ROOTS.get(distribution, frozenset({distribution.replace("-", "_")}))


@functools.cache
def _project_distributions() -> tuple[frozenset[str], frozenset[str]]:
    """Return the (base, optional) distribution names declared in pyproject."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    base = {_normalized_distribution_name(item) for item in project["dependencies"]}
    optional = {
        _normalized_distribution_name(item)
        for dependencies in project["optional-dependencies"].values()
        for item in dependencies
    }
    optional.discard(_normalized_distribution_name(project["name"]))
    return frozenset(base), frozenset(optional)


@functools.cache
def _optional_only_import_prefixes() -> frozenset[str]:
    """Resolve import prefixes supplied only by project optional dependencies."""
    base_distributions, optional_distributions = _project_distributions()
    base_prefixes = {prefix for item in base_distributions for prefix in _import_roots(item)}
    optional_prefixes = {prefix for item in optional_distributions for prefix in _import_roots(item)}
    return frozenset(optional_prefixes - base_prefixes)


def _optional_prefix_for(imported: str) -> str | None:
    """Return the optional-only prefix *imported* falls under, if any."""
    return next(
        (
            prefix
            for prefix in _optional_only_import_prefixes()
            if imported == prefix or imported.startswith(f"{prefix}.")
        ),
        None,
    )


class _ModuleScopeImportCollector(ast.NodeVisitor):
    """Collect static imports without entering runtime or type-checking scopes."""

    def __init__(self) -> None:
        self.imports: list[tuple[str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_If(self, node: ast.If) -> None:
        if _qualified_name(node.test) in {"TYPE_CHECKING", "typing.TYPE_CHECKING"}:
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend((alias.name, node.lineno) for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None:
            self.imports.append((node.module, node.lineno))


def _source_module_parts(path: Path) -> tuple[str, ...]:
    """Return the importable module parts for one source path."""
    absolute = path if path.is_absolute() else REPO_ROOT / path
    relative = absolute.relative_to(REPO_ROOT / "src").with_suffix("")
    parts = relative.parts
    return parts[:-1] if parts and parts[-1] == "__init__" else parts


def _resolved_import_from_base(path: Path, node: ast.ImportFrom) -> str | None:
    """Resolve an absolute or relative ``from`` import to its base module."""
    if node.level == 0:
        return node.module
    module_parts = _source_module_parts(path)
    absolute = path if path.is_absolute() else REPO_ROOT / path
    package_parts = module_parts if absolute.name == "__init__.py" else module_parts[:-1]
    ascend = node.level - 1
    if ascend > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - ascend]
    if node.module is not None:
        base_parts = (*base_parts, *node.module.split("."))
    return ".".join(base_parts) or None


def _first_party_module_exists(module: str) -> bool:
    target = REPO_ROOT / "src" / Path(*module.split("."))
    return target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()


class _ResolvedModuleScopeImportCollector(_ModuleScopeImportCollector):
    """Collect module-scope imports with first-party submodules resolved."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = _resolved_import_from_base(self._path, node)
        if base is None:
            return
        self.imports.append((base, node.lineno))
        self.imports.extend(
            (candidate, node.lineno)
            for alias in node.names
            if alias.name != "*"
            if _first_party_module_exists(candidate := f"{base}.{alias.name}")
        )


def _assert_optional_extra_imports_are_function_scoped(sources: Mapping[Path, str]) -> None:
    """Keep optional extras out of import-time dependencies below the app tier."""
    violations: list[str] = []
    for path, source in sources.items():
        if not any(path.is_relative_to(root) for root in _LOWER_TIER_ROOTS):
            continue
        if path in _OPTIONAL_IMPORT_ALLOWLIST:
            continue
        collector = _ModuleScopeImportCollector()
        collector.visit(_tree(path, source))
        for imported, line in collector.imports:
            prefix = _optional_prefix_for(imported)
            if prefix is not None:
                violations.append(
                    f"{path}:{line}: optional-extra-module-scope-imports forbids module-scope import of optional-only "
                    f"package {prefix!r}; violates AGENTS.md:3 (pyproject.toml is the source of truth for deps). "
                    "Fix: move the import inside the runtime function under try/except ImportError, or make the "
                    "dependency a project base dependency"
                )
    assert violations == [], "\n".join(violations)


def _assert_no_scroll_relative(sources: Mapping[Path, str]) -> None:
    violations = [
        f"{path}:{node.lineno}: use _simulate_chat_panel_user_scroll_y instead of scroll_relative"
        for path, source in sources.items()
        for node in ast.walk(_tree(path, source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "scroll_relative"
    ]
    assert violations == [], "\n".join(violations)


def _assert_no_unapproved_local_polling_helpers(sources: Mapping[Path, str]) -> None:
    """Ban today's polling identifiers, not structurally similar clones."""
    violations: list[str] = []
    for path, source in sources.items():
        for node in ast.walk(_tree(path, source)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in _POLLING_HELPER_NAMES:
                continue
            if path == Path("tests/support/waiting.py"):
                continue
            if (path, node.name) in _LOCAL_POLLING_ALLOWLIST:
                continue
            violations.append(f"{path}:{node.lineno}: use tests/support/waiting.py instead of local {node.name}")
    assert violations == [], "\n".join(violations)


_WAIT_CALLABLE = "wait_callable"
_WAIT_MODULE = "wait_module"
_WAIT_PACKAGE = "wait_package"
_OTHER_BINDING = "other"
_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _wait_scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Walk statements evaluated in one lexical scope, excluding nested scopes."""
    if isinstance(scope, ast.Module):
        stack: list[ast.AST] = list(reversed(scope.body))
    elif isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        stack = list(reversed(scope.body))
    elif isinstance(scope, ast.Lambda):
        stack = [scope.body]
    else:  # pragma: no cover - callers construct only the scope kinds above
        raise TypeError(f"unsupported scope: {type(scope).__name__}")
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _SCOPE_BOUNDARY_NODES):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _bound_names(target: ast.AST) -> set[str]:
    """Return identifiers bound by an assignment-like target."""
    return {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }


def _match_bound_names(pattern: ast.pattern) -> set[str]:
    """Return identifiers captured by a structural-pattern target."""
    names: set[str] = set()
    for node in ast.walk(pattern):
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name is not None:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            names.add(node.rest)
    return names


class _WaitUntilScopes:
    """Resolve wait helper imports through Python's enclosing function scopes."""

    def __init__(self, tree: ast.Module, path: Path) -> None:
        self.tree = tree
        self._path = path
        self.scopes: list[ast.AST] = [tree]
        self._parents: dict[int, ast.AST | None] = {id(tree): None}
        self._collect_scopes(tree, tree)
        self._globals: dict[int, set[str]] = {}
        self._nonlocals: dict[int, set[str]] = {}
        self._bindings: dict[int, dict[str, set[str]]] = {}
        for scope in self.scopes:
            nodes = tuple(_wait_scope_nodes(scope))
            self._globals[id(scope)] = {name for node in nodes if isinstance(node, ast.Global) for name in node.names}
            self._nonlocals[id(scope)] = {
                name for node in nodes if isinstance(node, ast.Nonlocal) for name in node.names
            }
            self._bindings[id(scope)] = self._scope_bindings(scope, nodes)
        self._relocate_declared_bindings()

    def _collect_scopes(self, node: ast.AST, parent: ast.AST) -> None:
        if isinstance(node, _FUNCTION_SCOPES):
            self.scopes.append(node)
            self._parents[id(node)] = parent
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                signature_nodes: list[ast.AST] = [node.args, *node.decorator_list]
                if node.returns is not None:
                    signature_nodes.append(node.returns)
                signature_nodes.extend(node.type_params)
                for signature_node in signature_nodes:
                    self._collect_scopes(signature_node, parent)
                for statement in node.body:
                    self._collect_scopes(statement, node)
            else:
                self._collect_scopes(node.body, node)
            return
        if isinstance(node, ast.ClassDef):
            # A class namespace is not an enclosing lexical scope for its methods.
            for child in (*node.decorator_list, *node.bases, *node.keywords, *node.type_params, *node.body):
                self._collect_scopes(child, parent)
            return
        for child in ast.iter_child_nodes(node):
            self._collect_scopes(child, parent)

    @staticmethod
    def _arguments(scope: ast.AST) -> set[str]:
        if not isinstance(scope, _FUNCTION_SCOPES):
            return set()
        arguments = scope.args
        return {
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *((arguments.vararg,) if arguments.vararg is not None else ()),
                *((arguments.kwarg,) if arguments.kwarg is not None else ()),
            )
        }

    def _import_bindings(self, node: ast.Import | ast.ImportFrom) -> Iterator[tuple[str, str]]:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if alias.name == "tests.support.waiting":
                    yield bound, _WAIT_MODULE if alias.asname else _WAIT_PACKAGE
                elif alias.asname is None and (alias.name == "tests" or alias.name.startswith("tests.")):
                    # Every unaliased dotted import binds the same top-level
                    # ``tests`` package; importing a sibling must not obscure
                    # the waiting module imported through that package.
                    yield bound, _WAIT_PACKAGE
                else:
                    yield bound, _OTHER_BINDING
            return
        module = _resolved_import_module(self._path, node)
        for alias in node.names:
            bound = alias.asname or alias.name
            if module == "tests.support.waiting" and alias.name == "wait_until":
                yield bound, _WAIT_CALLABLE
            elif module == "tests.support" and alias.name == "waiting":
                yield bound, _WAIT_MODULE
            else:
                yield bound, _OTHER_BINDING

    def _scope_bindings(self, scope: ast.AST, nodes: Sequence[ast.AST]) -> dict[str, set[str]]:
        bindings: dict[str, set[str]] = {}

        def record(names: set[str], origin: str = _OTHER_BINDING) -> None:
            for name in names:
                bindings.setdefault(name, set()).add(origin)

        record(self._arguments(scope))
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                record({node.name})
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for name, origin in self._import_bindings(node):
                    record({name}, origin)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    record(_bound_names(target))
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
                record(_bound_names(node.target))
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                record(_bound_names(node.optional_vars))
            elif isinstance(node, ast.ExceptHandler) and node.name is not None:
                record({node.name})
            elif isinstance(node, ast.Match):
                for case in node.cases:
                    record(_match_bound_names(case.pattern))
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    record(_bound_names(target))
        return bindings

    def _relocate_declared_bindings(self) -> None:
        """Apply ``global``/``nonlocal`` declarations to collected bindings."""
        module_bindings = self._bindings[id(self.tree)]
        for scope in self.scopes[1:]:
            bindings = self._bindings[id(scope)]
            for name in self._globals[id(scope)]:
                origins = bindings.pop(name, set())
                if origins:
                    module_bindings.setdefault(name, set()).update(origins)
            for name in self._nonlocals[id(scope)]:
                origins = bindings.pop(name, set())
                if not origins:
                    continue
                parent = self._parents[id(scope)]
                while parent is not None and parent is not self.tree:
                    if name in self._bindings[id(parent)] and name not in self._globals[id(parent)]:
                        self._bindings[id(parent)][name].update(origins)
                        break
                    parent = self._parents[id(parent)]

    def _resolved_origins(self, scope: ast.AST, name: str) -> set[str]:
        current: ast.AST | None = self.tree if name in self._globals[id(scope)] else scope
        if name in self._nonlocals[id(scope)]:
            current = self._parents[id(scope)]
        while current is not None:
            origins = self._bindings[id(current)].get(name)
            if origins is not None:
                return origins
            current = self._parents[id(current)]
        return set()

    def resolves_shared_wait_until(self, scope: ast.AST, callee: ast.expr) -> bool:
        parts = _qualified_name(callee).split(".")
        if len(parts) == 1:
            expected = _WAIT_CALLABLE
        elif len(parts) == 2 and parts[1] == "wait_until":
            expected = _WAIT_MODULE
        elif len(parts) == 4 and parts[1:] == ["support", "waiting", "wait_until"]:
            expected = _WAIT_PACKAGE
        else:
            return False
        return self._resolved_origins(scope, parts[0]) == {expected}


def _assert_no_ignored_wait_until_results(sources: Mapping[Path, str]) -> None:
    """A positive wait must surface timeout instead of discarding ``False``."""
    violations: list[str] = []
    for path, source in sources.items():
        tree = _tree(path, source)
        scopes = _WaitUntilScopes(tree, path)
        for scope in scopes.scopes:
            for node in _wait_scope_nodes(scope):
                if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Await):
                    continue
                call = node.value.value
                if not isinstance(call, ast.Call) or not scopes.resolves_shared_wait_until(scope, call.func):
                    continue
                violations.append(
                    f"{path}:{node.lineno}: ignored wait_until result hides timeout; "
                    "use wait_for(...) or assert/branch on the bool result"
                )
    assert violations == [], "\n".join(violations)


def _assert_integration_marker_directory_disjoint(sources: Mapping[Path, str]) -> None:
    violations: list[str] = []
    for path, source in sources.items():
        if not path.is_relative_to(Path("tests/integration")):
            continue
        for node in ast.walk(_tree(path, source)):
            if isinstance(node, ast.Attribute) and _qualified_name(node) == "pytest.mark.integration":
                violations.append(
                    f"{path}:{node.lineno}: tests/integration contains offline cross-layer tests; "
                    "do not mark them integration"
                )
    assert violations == [], "\n".join(violations)


def _agent_engine_constructor_refs(tree: ast.Module) -> set[str]:
    """Return the names by which top-level imports expose AgentEngine."""
    refs = {"AgentEngine"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                module_ref = imported.asname or imported.name
                refs.add(f"{module_ref}.AgentEngine")
        elif isinstance(node, ast.ImportFrom):
            refs.update(imported.asname or imported.name for imported in node.names if imported.name == "AgentEngine")
    return refs


def _assigned_engine_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    constructor_refs: set[str],
) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or _qualified_name(value.func) not in constructor_refs:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _direct_engine_start_lines(tree: ast.Module) -> list[int]:
    """Return the lines where a locally constructed AgentEngine is started.

    Shared with the allowlist pin so an exemption stays tied to the exact
    construct it exempts: a file that merely builds an engine no longer earns
    one, and the pin cannot certify an entry the guard would never have flagged.
    """
    lines: list[int] = []
    constructor_refs = _agent_engine_constructor_refs(tree)
    for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        engine_names = _assigned_engine_names(function, constructor_refs)
        lines.extend(
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "start"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in engine_names
        )
    return sorted(lines)


def _assert_no_direct_agent_engine_start(sources: Mapping[Path, str]) -> None:
    violations = [
        f"{path}:{line}: construct started AgentEngine instances with the agent_engine fixture"
        for path, source in sources.items()
        if path not in _ENGINE_START_PATH_ALLOWLIST
        for line in _direct_engine_start_lines(_tree(path, source))
    ]
    assert violations == [], "\n".join(violations)


def _assert_quarantine_marker_metadata(sources: Mapping[Path, str], *, today: date | None = None) -> None:
    """Validate quarantine metadata statically — including expiry.

    Expiry must be enforced here, not only at test setup: CI's marker
    expression deselects integration/gc_calibration tests and the collect-only
    prewarm never runs setup, so a quarantine on those tests would otherwise
    outlive its expiry without ever failing CI. The bare (uncalled)
    ``@pytest.mark.quarantine`` spelling is rejected for the same reason — it
    is valid pytest but carries no metadata, and on deselected tests the
    setup-time check would never see it.
    """
    current_date = today or datetime.now(UTC).date()
    violations: list[str] = []
    for path, source in sources.items():
        tree = _tree(path, source)
        call_funcs = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and id(node) not in call_funcs
                and _qualified_name(node) == "pytest.mark.quarantine"
            ):
                violations.append(
                    f"{path}:{node.lineno}: bare quarantine marker carries no metadata; "
                    "call it with reason= and expires='YYYY-MM-DD'"
                )
                continue
            if not isinstance(node, ast.Call) or _qualified_name(node.func) != "pytest.mark.quarantine":
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None}
            reason = keywords.get("reason")
            expires = keywords.get("expires")
            if not isinstance(reason, ast.Constant) or not isinstance(reason.value, str) or not reason.value.strip():
                violations.append(f"{path}:{node.lineno}: quarantine requires a non-empty reason=")
            if not isinstance(expires, ast.Constant) or not isinstance(expires.value, str):
                violations.append(f"{path}:{node.lineno}: quarantine requires expires='YYYY-MM-DD'")
                continue
            try:
                parsed = date.fromisoformat(expires.value)
            except ValueError:
                violations.append(f"{path}:{node.lineno}: invalid quarantine expiry {expires.value!r}; use YYYY-MM-DD")
                continue
            if parsed.isoformat() != expires.value:
                violations.append(f"{path}:{node.lineno}: invalid quarantine expiry {expires.value!r}; use YYYY-MM-DD")
            elif parsed < current_date:
                violations.append(
                    f"{path}:{node.lineno}: quarantine expired on {expires.value}; "
                    "remove it, extend it with justification, or fix the test"
                )
    assert violations == [], "\n".join(violations)


def _raw_tool_loop_layer_aliases(tree: ast.Module) -> set[str]:
    """Local names bound to the raw ToolLoopLayer by imports, aliases included."""
    names = {"ToolLoopLayer"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "ToolLoopLayer" and alias.asname:
                    names.add(alias.asname)
    return names


def _raw_tool_loop_layer_uses(tree: ast.Module) -> list[tuple[int, str | None, str]]:
    """(line, scope, kind) per raw-layer use.

    Constructions are attributed to their nearest enclosing function and
    subclass bases to the class being defined, so an allowlist entry vouches
    only for the exact scope it names.
    """
    aliases = _raw_tool_loop_layer_aliases(tree)

    def _is_raw_layer_ref(node: ast.expr) -> bool:
        name = _qualified_name(node)
        return name in aliases or name.endswith(".ToolLoopLayer")

    found: list[tuple[int, str | None, str]] = []

    def _walk(node: ast.AST, enclosing: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call) and _is_raw_layer_ref(child.func):
                found.append((child.lineno, enclosing, "construction"))
            if isinstance(child, ast.ClassDef) and any(_is_raw_layer_ref(base) for base in child.bases):
                found.append((child.lineno, child.name, "subclass"))
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _walk(child, child.name)
            else:
                _walk(child, enclosing)

    _walk(tree, None)
    return found


def _assert_tool_loop_layer_uses_are_invariant_checked(sources: Mapping[Path, str]) -> None:
    """Loop transcripts are only guarded if the invariant oracle sees them.

    A bare ``ToolLoopLayer`` in a test — constructed directly (under any
    import alias) or used as a base class — silently opts final responses out
    of the transcript-invariant oracle
    (``tests.support.transcript_invariants``); tests must build or derive
    from ``InvariantCheckedToolLoopLayer`` instead so the oracle's coverage
    claim stays mechanically true. Only sites that never produce a
    loop-landed final response (constructor pins, the degenerate-stream
    fallback) plus the checked layer's own definition may use the raw layer,
    each via an explicit allowlist entry naming its scope.
    """
    violations: list[str] = []
    for path, source in sources.items():
        for lineno, scope, kind in _raw_tool_loop_layer_uses(_tree(path, source)):
            if (path, scope) in _RAW_TOOL_LOOP_LAYER_ALLOWLIST:
                continue
            action = "build" if kind == "construction" else "subclass"
            violations.append(
                f"{path}:{lineno}: {action} InvariantCheckedToolLoopLayer "
                "(tests.support.transcript_invariants) so the transcript-invariant oracle checks "
                "final responses; a raw ToolLoopLayer needs an allowlist entry"
            )
    assert violations == [], "\n".join(violations)


def _assert_trajectory_prefix_checks_use_physical_slots(sources: Mapping[Path, str]) -> None:
    """Route trajectory-file assertions through the slots-only shared oracle."""
    violations = [
        f"{path}:{node.lineno}: use tests.support.trajectory_invariants.assert_trajectory_accounted "
        "so the prefix check sees physical slots"
        for path, source in sources.items()
        if path not in _DIRECT_TRAJECTORY_PREFIX_ORACLE_ALLOWLIST
        for node in ast.walk(_tree(path, source))
        if isinstance(node, ast.Call) and _qualified_name(node.func).endswith("verify_accounted_prefix")
    ]
    assert violations == [], "\n".join(violations)


# --- exchange-walker guard (scans src/chrys, not tests) ---------------------
#
# The transcript exchange grammar lives in chrys.kernel.exchanges
# (iter_exchanges/pair_results); every production consumer reads boundaries and
# pairing from it. This guard is the tripwire against the NEXT hand-rolled
# walker: transcript loops that re-derive exchange structure locally. Tests are
# deliberately out of scope — the invariant oracle and the differential
# reference machine are independent walkers BY DESIGN.

_EXCHANGE_WALKER_ALLOWLIST = {
    # Annotation-occurrence scanner: walks persisted group-annotation runs, not
    # exchanges; result classification only delimits occurrence ends.
    (Path("src/chrys/kernel/compaction.py"), "_partial_tool_call_groups"),
    # Incremental re-annotation rewind: walks BACKWARD along the grammar's
    # member shapes to the enclosing exchange's start so regrouping sees the
    # whole exchange (call/result fusion) without rescanning the prefix —
    # running iter_exchanges forward would cost the full-list pass this
    # rewind exists to avoid. Makes no pairing decision; grouping itself
    # still consumes iter_exchanges output.
    (Path("src/chrys/kernel/compaction.py"), "_reannotation_start"),
    # Group-level pairing consumers: operate WITHIN one already-annotated
    # group, with deliberately local policies (preservation-pinned).
    (Path("src/chrys/service/context/compaction/scoped.py"), "_tool_group_integrity"),
    (Path("src/chrys/service/context/compaction/last_words.py"), "_format_tool_group"),
    (Path("src/chrys/service/context/compaction/summaries.py"), "_build_summary"),
    (Path("src/chrys/service/llm/openai_responses.py"), "RawOpenAIChatClient._classify_reasoning_replay_groups"),
    # Display-metadata fold-range scan; renders summary chrome, pairs nothing.
    (Path("src/chrys/service/context/providers/history.py"), "_auto_summary"),
    # Positional current-turn slot bucketing for batch-id stamping — a
    # (call_id, name) presentation scan, not an exchange walk.
    (Path("src/chrys/service/session/history.py"), "SessionHistoryManager.persist_batch_ids"),
    # Pairs file-tool calls positionally with edit-snapshot refs over
    # serialized dicts; fail-soft zip truncation is deliberate.
    (Path("src/chrys/app/tui/screens/main/session_handlers.py"), "SessionHandler.load_file_edit_snapshots"),
    # Deliberately NARROW legacy-sidecar dedup policy over batch-tagged
    # tool-call messages; justified with a preservation pin, not migrated.
    (Path("src/chrys/app/tui/widgets/chat/replay.py"), "_legacy_duplicate_intermediate_sidecars"),
    # Replay-ID minting totals over the globally-coerced presentation key;
    # pairing decisions live in the grammar-backed coordinate map.
    # Per-content image-stub rewrite: rebuilds function results around
    # stubbed images (falsy call ids normalize to ""); makes no pairing or
    # boundary decision.
    (Path("src/chrys/service/vision.py"), "NonVisionImageStubMiddleware.process"),
    # Bounded fallback-timeline renderer over already-scoped groups; the
    # sibling group formatter it dispatches to is display-only pairing.
    (Path("src/chrys/service/context/compaction/last_words.py"), "_format_dropped"),
    # Wire serializers: map each message to provider payload and flatten;
    # ids are echoed verbatim, pairing rides the transcript unchanged.
    (Path("src/chrys/service/llm/deepseek.py"), "DeepSeekChatCompletionClient._prepare_messages_for_openai"),
    (
        Path("src/chrys/service/llm/openai_chat_completion.py"),
        "RawOpenAIChatCompletionClient._prepare_messages_for_openai",
    ),
}

# Outside the grammar module, importing the role-gated result-only classifier
# is itself a walker smell: consumers should consume iter_exchanges output,
# not re-classify messages. Justified importers only.
_CLASSIFIER_IMPORT_ALLOWLIST = {
    # The group annotator and the annotation-occurrence scanner share the
    # grammar's follower rule for their secondary, compaction-owned partition.
    Path("src/chrys/kernel/compaction.py"),
}

_EXCHANGE_GRAMMAR_MODULE = Path("src/chrys/kernel/exchanges.py")
_RESULT_ONLY_CLASSIFIER = "is_result_only_message"
_EXCHANGE_GRAMMAR_ENTRYPOINTS = {"iter_exchanges", "pair_results"}
_TYPE_SET_NAMES = ("TOOL_CALL_CONTENT_TYPES", "TOOL_RESULT_CONTENT_TYPES")
_TOOL_CONTENT_TYPE_STRINGS = frozenset(
    {
        "function_call",
        "hosted_tool_call",
        "code_interpreter_tool_call",
        "image_generation_tool_call",
        "mcp_server_tool_call",
        "search_tool_call",
        "shell_tool_call",
        "function_result",
        "hosted_tool_result",
        "code_interpreter_tool_result",
        "image_generation_tool_result",
        "mcp_server_tool_result",
        "search_tool_result",
        "shell_tool_result",
        "legacy",
    }
)
_LOOP_NODES = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_COMPREHENSION_NODES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SCOPE_BOUNDARY_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _scoped_walk(root: ast.AST) -> Iterator[ast.AST]:
    """``ast.walk`` confined to the root's own scope.

    A nested function, lambda, or class body is a separate scope whose
    cursors and reads get their own guard evaluation; attributing them to
    the enclosing function would flag a formatter for a helper it merely
    defines.
    """
    stack: list[ast.AST] = [root]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, _SCOPE_BOUNDARY_NODES):
                stack.append(child)


def _loop_target_names(func: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in _scoped_walk(func):
        targets: list[ast.expr] = []
        if isinstance(node, (ast.For, ast.AsyncFor)):
            targets.append(node.target)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            targets.extend(generator.target for generator in node.generators)
        for target in targets:
            names.update(name.id for name in ast.walk(target) if isinstance(name, ast.Name))
    return names


def _walked_item_names(func: ast.AST) -> set[str]:
    """Loop targets plus simple aliases of them.

    ``message = messages[index]`` and ``message = item`` walk the same item
    as the loop construct itself; the fixpoint keeps chained rebindings from
    hiding a role read behind a fresh name.
    """
    names = _loop_target_names(func)
    changed = True
    while changed:
        changed = False
        for node in _scoped_walk(func):
            target: ast.expr | None = None
            value: ast.expr | None = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                target, value = node.target, node.value
            if not isinstance(target, ast.Name) or target.id in names or value is None:
                continue
            if isinstance(value, ast.Subscript) or (isinstance(value, ast.Name) and value.id in names):
                names.add(target.id)
                changed = True
    return names


def _role_read_bases(node: ast.AST) -> list[ast.expr]:
    """The expression each ``role`` read is performed on, in any spelling."""
    bases: list[ast.expr] = []
    for child in _scoped_walk(node):
        if isinstance(child, ast.Attribute) and child.attr == "role":
            bases.append(child.value)
        elif (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "get"
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and child.args[0].value == "role"
        ):
            bases.append(child.func.value)
        elif isinstance(child, ast.Subscript) and isinstance(child.slice, ast.Constant) and child.slice.value == "role":
            bases.append(child.value)
    return bases


def _reads_walked_role(node: ast.AST, walked_names: set[str], matches_role_helper: Callable[[ast.expr], bool]) -> bool:
    """Role read off a loop-walked item — a loop-target name or a subscript.

    Reading role off a plain parameter (a per-message wire serializer) is not
    a transcript walk; reading it off the item a loop iterates, or off
    ``messages[index]`` in a cursor loop, is. Passing a walked item to a
    local role-reading helper counts the same as reading inline.
    """

    def _is_walked(value: ast.expr) -> bool:
        if isinstance(value, ast.Name) and value.id in walked_names:
            return True
        return isinstance(value, ast.Subscript)

    if any(_is_walked(base) for base in _role_read_bases(node)):
        return True
    return any(
        isinstance(child, ast.Call)
        and matches_role_helper(child.func)
        and any(_is_walked(argument) for argument in [*child.args, *(keyword.value for keyword in child.keywords)])
        for child in _scoped_walk(node)
    )


def _classification_references(node: ast.AST, matches_classifier: Callable[[ast.expr], bool]) -> bool:
    """Tool call/result classification: type sets, type strings, classifiers.

    Helper indirection does not launder a walker: calling a locally defined
    classifier (any local helper that itself references classification)
    counts the same as reading the type sets inline.
    """
    for child in _scoped_walk(node):
        if isinstance(child, (ast.Name, ast.Attribute)):
            name = _qualified_name(child)
            if name.endswith(_TYPE_SET_NAMES) or name.split(".")[-1] == _RESULT_ONLY_CLASSIFIER:
                return True
        if isinstance(child, ast.Constant) and child.value in _TOOL_CONTENT_TYPE_STRINGS:
            return True
        if isinstance(child, ast.Call) and matches_classifier(child.func):
            return True
    return False


def _reads_tool_id(node: ast.AST) -> bool:
    """A call_id/image_id read in any spelling."""
    for child in _scoped_walk(node):
        if isinstance(child, ast.Attribute) and child.attr in ("call_id", "image_id"):
            return True
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "get"
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and child.args[0].value in ("call_id", "image_id")
        ):
            return True
        if (
            isinstance(child, ast.Subscript)
            and isinstance(child.slice, ast.Constant)
            and child.slice.value in ("call_id", "image_id")
        ):
            return True
    return False


def _enumerate_target_index(target: ast.expr, source: ast.expr) -> str | None:
    """The index name a ``for index, item in enumerate(...)`` target binds,
    seeing through the identity-preserving container wrappers —
    ``reversed(list(enumerate(...)))`` is a backwards enumerate walk."""
    while (
        isinstance(source, ast.Call)
        and _qualified_name(source.func).split(".")[-1] in _CONTAINER_WRAPPER_CALLEES
        and source.args
    ):
        source = source.args[0]
    if (
        isinstance(source, ast.Call)
        and _qualified_name(source.func) == "enumerate"
        and isinstance(target, ast.Tuple)
        and target.elts
        and isinstance(target.elts[0], ast.Name)
    ):
        return target.elts[0].id
    return None


def _enumerate_index_names(func: ast.AST) -> set[str]:
    """Index targets of statement-level enumerate loops.

    Comprehension targets are Python-scoped to the comprehension and are
    judged per comprehension node, never pooled into the function scope —
    an outer name that happens to match one is not a cursor.
    """
    names: set[str] = set()
    for node in _scoped_walk(func):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            index = _enumerate_target_index(node.target, node.iter)
            if index is not None:
                names.add(index)
    return names


def _comprehension_enumerate_indices(comp_node: ast.AST) -> set[str]:
    """Enumerate indices bound by one comprehension's own generators."""
    names: set[str] = set()
    for generator in getattr(comp_node, "generators", []):
        index = _enumerate_target_index(generator.target, generator.iter)
        if index is not None:
            names.add(index)
    return names


def _contains_name(node: ast.AST, names: set[str]) -> bool:
    return any(isinstance(child, ast.Name) and child.id in names for child in ast.walk(node))


def _is_membership_test(node: ast.AST) -> bool:
    return isinstance(node, ast.Compare) and all(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)


def _captures_name(node: ast.AST, names: set[str]) -> bool:
    """A membership test asks a question about the index, an f-string
    renders it as text, and a comprehension's value slot collects results
    into a container (the comprehension spelling of append-reporting) —
    none of them capture its value. Comprehension filters and iterables
    still do. Equality/order comparisons are boundary logic and get their
    own construct rule."""
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if _is_membership_test(current) or isinstance(current, ast.JoinedStr):
            continue
        if isinstance(current, _COMPREHENSION_NODES):
            for generator in current.generators:
                stack.append(generator.iter)
                stack.extend(generator.ifs)
            continue
        if isinstance(current, ast.Name) and current.id in names:
            return True
        stack.extend(ast.iter_child_nodes(current))
    return False


def _capture_slots(target: ast.expr) -> list[ast.expr]:
    """The assignable slots of a target, with tuple/list unpacking flattened."""
    if isinstance(target, (ast.Tuple, ast.List)):
        return [slot for element in target.elts for slot in _capture_slots(element)]
    if isinstance(target, ast.Starred):
        return _capture_slots(target.value)
    return [target]


def _is_literal_expression(node: ast.AST) -> bool:
    """A literal in any AST spelling — nothing resolved at runtime."""
    return not any(isinstance(child, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)) for child in ast.walk(node))


def _is_scalar_capture_slot(slot: ast.expr) -> bool:
    """Names, attributes, and FIXED subscript keys hold cursor state; a
    dynamic subscript key is the coordinate/reporter-map shape."""
    if isinstance(slot, (ast.Name, ast.Attribute)):
        return True
    return isinstance(slot, ast.Subscript) and _is_literal_expression(slot.slice)


def _fixed_pairs_capture(node: ast.AST, names: set[str]) -> bool:
    """A mapping-shaped expression writing a captured value under a literal
    key, in any ordinary ``dict.update`` input spelling: a dict literal, a
    ``dict(...)`` call, or an iterable of ``(key, value)`` pairs."""
    if isinstance(node, ast.Dict):
        return any(
            key is not None and _is_literal_expression(key) and _captures_name(value, names)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    if isinstance(node, ast.Call) and _qualified_name(node.func).split(".")[-1] == "dict":
        return any(
            _captures_name(keyword.value, names)
            if keyword.arg is not None
            else _fixed_pairs_capture(keyword.value, names)
            for keyword in node.keywords
        ) or any(_fixed_pairs_capture(argument, names) for argument in node.args)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(
            isinstance(element, ast.Tuple)
            and len(element.elts) == 2
            and _is_literal_expression(element.elts[0])
            and _captures_name(element.elts[1], names)
            for element in node.elts
        )
    return False


def _is_mapping_shaped(node: ast.AST) -> bool:
    """An argument whose SHAPE proves dict.update mapping semantics — with
    one present, sibling keyword arguments are dict fields, not API params.
    A ``|`` union is mapping-shaped when either side is."""
    if isinstance(node, ast.Dict):
        return True
    if isinstance(node, ast.Call) and _qualified_name(node.func).split(".")[-1] == "dict":
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return bool(node.elts) and all(
            isinstance(element, ast.Tuple) and len(element.elts) == 2 for element in node.elts
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_mapping_shaped(node.left) or _is_mapping_shaped(node.right)
    return False


_SCALAR_COLLAPSE_CALLEES = ("max", "min", "next", "sum")
_CONTAINER_WRAPPER_CALLEES = ("list", "sorted", "tuple", "reversed", "iter")


def _is_fixed_field_store_call(call: ast.Call, names: set[str]) -> bool:
    """update/setdefault/setattr are the call spellings of a fixed-key store.

    ``bounds.update(end=index)``, ``bounds.update({"end": index})`` (in any
    mapping spelling) and ``bounds.setdefault("end", index)`` /
    ``setattr(bounds, "end", index)`` write cursor state exactly like
    ``bounds["end"] = index``. A dynamic key keeps the coordinate-map
    exemption; value-passing callees (emit, log, append, insert) hand the
    index to another component and stay reporting — as does an ``update``
    call carrying a positional handle argument (``progress.update(task_id,
    completed=position)``), which is the reporter-API shape rather than the
    keyword-only or mapping-shaped dict.update idioms.
    """
    callee = _qualified_name(call.func).split(".")[-1]
    if callee == "update":
        keyword_capture = any(
            _captures_name(keyword.value, names)
            if keyword.arg is not None
            else _fixed_pairs_capture(keyword.value, names)
            for keyword in call.keywords
        )
        if keyword_capture and (not call.args or any(_is_mapping_shaped(argument) for argument in call.args)):
            return True
        return any(_fixed_pairs_capture(argument, names) for argument in call.args)
    if callee == "setdefault":
        return (
            len(call.args) >= 2
            and _is_literal_expression(call.args[0])
            and any(_captures_name(argument, names) for argument in call.args[1:])
        )
    if callee == "setattr":
        return len(call.args) >= 3 and _is_literal_expression(call.args[1]) and _captures_name(call.args[2], names)
    return False


def _comprehension_operand(node: ast.AST) -> ast.AST | None:
    """The comprehension an expression operates on, unwrapping the
    identity-preserving container wrappers (list/sorted/tuple/reversed/iter)."""
    while (
        isinstance(node, ast.Call)
        and _qualified_name(node.func).split(".")[-1] in _CONTAINER_WRAPPER_CALLEES
        and node.args
    ):
        node = node.args[0]
    return node if isinstance(node, _COMPREHENSION_NODES) else None


def _unpacked_scalar_slots(target: ast.expr) -> list[ast.expr]:
    """Non-starred slots a destructuring target extracts as scalars.

    A plain name binds the whole container (a positions report keeps
    reporting); ``end, = ...`` extracts the element. Starred slots bind
    lists and stay containers."""
    if isinstance(target, (ast.Tuple, ast.List)):
        return [
            slot
            for element in target.elts
            if not isinstance(element, ast.Starred)
            for slot in (
                [element] if not isinstance(element, (ast.Tuple, ast.List)) else _unpacked_scalar_slots(element)
            )
        ]
    return []


def _comprehension_collapses_index(node: ast.AST) -> bool:
    """A comprehension (possibly wrapper-wrapped) whose value slot captures
    its OWN enumerate index — collapsing it yields scalar cursor state."""
    comp = _comprehension_operand(node)
    if comp is None:
        return False
    indices = _comprehension_enumerate_indices(comp)
    return bool(indices) and any(_captures_name(value, indices) for value in _comprehension_values(comp))


def _comprehension_values(node: ast.AST) -> list[ast.expr]:
    """The value slots a comprehension collects into its container."""
    if isinstance(node, ast.DictComp):
        return [node.key, node.value]
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return [node.elt]
    return []


def _has_pairing_construct(func: ast.AST, matches_id_reader: Callable[[ast.expr], bool]) -> bool:
    """Pairing/boundary state: id reads, index arithmetic, or a cursor walk.

    A pure content formatter renders types without any of these and stays
    outside the guard's reach. The id read is deliberately coarse — a
    transcript-looping serializer that only ECHOES call ids still matches,
    and earns a justified allowlist entry rather than a narrower trigger:
    under-triggering here is what lets a genuinely new walker ship. A call
    to a local id-reading helper counts the same as reading inline.
    An enumerate index is cursor bookkeeping when its VALUE is captured
    (arithmetic, subscripting by it, returning/yielding it, storing it
    into a name, attribute, or fixed subscript key — through tuple
    unpacking or a fixed-field store call like update/setdefault/setattr)
    or when it GATES control flow (an equality/order comparison, an
    if/while/conditional-expression test, a match subject or case guard).
    Comprehension targets are judged within their own comprehension: a
    filter that gates on the index, or a collapse of an index-valued
    comprehension to a scalar (max/min/next/sum, or a non-slice subscript,
    through identity-preserving wrappers), is cursor bookkeeping there.
    Membership tests, f-string rendering, append-style and keyword
    reporting, comprehension container building (returned or sliced), and
    dynamic-key coordinate maps carry no pairing state; a nested function
    or lambda's cursor belongs to its own scope.
    """
    if _reads_tool_id(func):
        return True
    enumerate_indices = _enumerate_index_names(func)
    for child in _scoped_walk(func):
        if isinstance(child, ast.Call) and matches_id_reader(child.func):
            return True
        if isinstance(child, ast.Subscript) and any(isinstance(node, ast.BinOp) for node in ast.walk(child.slice)):
            return True
        if isinstance(child, ast.While) and any(
            isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Name) for node in _scoped_walk(child)
        ):
            return True
        if (
            isinstance(child, ast.For)
            and isinstance(child.iter, ast.Call)
            and _qualified_name(child.iter.func) == "range"
            and any(
                isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Name) for node in _scoped_walk(child)
            )
        ):
            return True
        if isinstance(child, _COMPREHENSION_NODES):
            comp_indices = _comprehension_enumerate_indices(child)
            if comp_indices and any(
                _captures_name(test, comp_indices) for generator in child.generators for test in generator.ifs
            ):
                return True
        if (
            isinstance(child, ast.Call)
            and _qualified_name(child.func).split(".")[-1] in _SCALAR_COLLAPSE_CALLEES
            and any(_comprehension_collapses_index(argument) for argument in child.args)
        ):
            return True
        if (
            isinstance(child, ast.Subscript)
            and not isinstance(child.slice, ast.Slice)
            and _comprehension_collapses_index(child.value)
        ):
            return True
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "pop"
            and _comprehension_collapses_index(child.func.value)
        ):
            return True
        if (
            isinstance(child, ast.Assign)
            and _comprehension_collapses_index(child.value)
            and any(
                _is_scalar_capture_slot(slot) for target in child.targets for slot in _unpacked_scalar_slots(target)
            )
        ):
            return True
        if not enumerate_indices:
            continue
        if isinstance(child, ast.BinOp) and _contains_name(child, enumerate_indices):
            return True
        if (
            isinstance(child, ast.Subscript)
            and isinstance(child.slice, ast.Name)
            and child.slice.id in enumerate_indices
        ):
            return True
        if (
            isinstance(child, ast.Compare)
            and not _is_membership_test(child)
            and _captures_name(child, enumerate_indices)
        ):
            return True
        if isinstance(child, (ast.If, ast.While, ast.IfExp)) and _captures_name(child.test, enumerate_indices):
            return True
        if isinstance(child, ast.Match) and _captures_name(child.subject, enumerate_indices):
            return True
        if (
            isinstance(child, ast.match_case)
            and child.guard is not None
            and _captures_name(child.guard, enumerate_indices)
        ):
            return True
        if isinstance(child, ast.Call) and _is_fixed_field_store_call(child, enumerate_indices):
            return True
        if (
            isinstance(child, (ast.Return, ast.Yield))
            and child.value is not None
            and _captures_name(child.value, enumerate_indices)
        ):
            return True
        capture_targets: list[ast.expr] = []
        capture_value: ast.expr | None = None
        if isinstance(child, ast.Assign):
            capture_targets, capture_value = child.targets, child.value
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            capture_targets, capture_value = [child.target], child.value
        if (
            capture_value is not None
            and any(_is_scalar_capture_slot(slot) for target in capture_targets for slot in _capture_slots(target))
            and _captures_name(capture_value, enumerate_indices)
        ):
            return True
    return False


def _references_exchange_grammar(func: ast.AST, matches_grammar_caller: Callable[[ast.expr], bool]) -> bool:
    """A grammar CALL whose result flows — naming or discarding is not consuming.

    Calling a local helper that itself calls the grammar counts too: a
    consumer may precompute pairing in one function and walk with the
    result in another. A bare expression-statement call throws its result
    away and exempts nothing.
    """
    discarded = {node.value for node in _scoped_walk(func) if isinstance(node, ast.Expr)}
    return any(
        isinstance(node, ast.Call)
        and node not in discarded
        and (
            _qualified_name(node.func).split(".")[-1] in _EXCHANGE_GRAMMAR_ENTRYPOINTS
            or matches_grammar_caller(node.func)
        )
        for node in _scoped_walk(func)
    )


class _ModuleScopes:
    """Lexical scope index for one module.

    Records every function-like scope (defs AND lambdas — each gets exactly
    one guard evaluation of its own) with its enclosing-scope chain,
    qualified name, and owning class, plus every named helper binding: def
    statements and name-bound lambdas alike. Helper calls resolve with
    lexical fidelity — a bare name walks the enclosing def scopes out to
    module level and the NEAREST binding shadows outer ones (class bodies
    are not name scopes), where a non-helper binding at a level (parameter,
    assignment, loop target) makes the name OPAQUE and stops resolution;
    ``self.``/``cls.`` attributes resolve to methods of the calling scope's
    class lineage (local base classes included); any other attribute base
    falls back to coarse last-name matching against module-level functions
    and methods, where a bare-name call could not resolve — strict callers
    demand ALL same-named candidates agree before trusting the fallback.
    """

    def __init__(self, tree: ast.Module) -> None:
        self.scopes: list[ast.AST] = []
        self.qualified: dict[int, str] = {}
        self._tree = tree
        self._chains: dict[int, tuple[ast.AST, ...]] = {}
        self._owners: dict[int, ast.ClassDef | None] = {}
        # name -> [(helper node, id of binding def scope or None, class the
        # binding hangs off when it is a direct class attribute)]
        self._helpers: dict[str, list[tuple[ast.AST, int | None, ast.ClassDef | None]]] = {}
        self._classes: dict[str, list[ast.ClassDef]] = {}
        self._opaque: dict[int | None, frozenset[str]] = {}
        self._lineages: dict[int, frozenset[int]] = {}
        self._collect(tree, (), "", None)

    def _add_helper(
        self, name: str, node: ast.AST, chain: tuple[ast.AST, ...], method_class: ast.ClassDef | None
    ) -> None:
        binding = id(chain[-1]) if chain else None
        self._helpers.setdefault(name, []).append((node, binding, method_class))

    def _collect(self, node: ast.AST, chain: tuple[ast.AST, ...], qual: str, owner: ast.ClassDef | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                name = getattr(child, "name", "<lambda>")
                self.scopes.append(child)
                self.qualified[id(child)] = qual + name
                self._chains[id(child)] = (*chain, child)
                self._owners[id(child)] = owner
                if not isinstance(child, ast.Lambda):
                    self._add_helper(child.name, child, chain, owner if isinstance(node, ast.ClassDef) else None)
                self._collect(child, (*chain, child), qual + name + ".", owner)
            elif isinstance(child, ast.ClassDef):
                self._classes.setdefault(child.name, []).append(child)
                self._collect(child, chain, qual + child.name + ".", child)
            else:
                if isinstance(child, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and isinstance(
                    child.value, ast.Lambda
                ):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            self._add_helper(
                                target.id, child.value, chain, owner if isinstance(node, ast.ClassDef) else None
                            )
                self._collect(child, chain, qual, owner)

    def _opaque_names(self, level: ast.AST | None) -> frozenset[str]:
        """Names a def scope (or the module) binds OUTSIDE the helper index.

        A parameter, assignment, or loop target rebinds the name to a value
        the index knows nothing about — a call through it must not resolve
        to an outer helper of the same name."""
        key = None if level is None else id(level)
        cached = self._opaque.get(key)
        if cached is not None:
            return cached
        names: set[str] = set()
        scope: ast.AST = self._tree if level is None else level
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = scope.args
            names.update(arg.arg for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs))
            names.update(arg.arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None)
        for child in _scoped_walk(scope):
            if isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
            targets: tuple[ast.expr, ...] = ()
            if isinstance(child, ast.Assign):
                targets = tuple(child.targets)
            elif isinstance(child, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
                targets = (child.target,)
            elif isinstance(child, ast.withitem) and child.optional_vars is not None:
                targets = (child.optional_vars,)
            for target in targets:
                names.update(
                    node.id
                    for node in ast.walk(target)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                )
        result = frozenset(names)
        self._opaque[key] = result
        return result

    def _lineage(self, owner: ast.ClassDef) -> frozenset[int]:
        """The class and its LOCAL ancestors — where ``self.helper`` can live."""
        cached = self._lineages.get(id(owner))
        if cached is not None:
            return cached
        seen: set[int] = set()
        queue = [owner]
        while queue:
            cls = queue.pop()
            if id(cls) in seen:
                continue
            seen.add(id(cls))
            for base in cls.bases:
                if isinstance(base, ast.Subscript):
                    base = base.value
                base_name = _qualified_name(base).split(".")[-1]
                queue.extend(self._classes.get(base_name, []))
        result = frozenset(seen)
        self._lineages[id(owner)] = result
        return result

    def resolves_flagged(
        self, caller: ast.AST, callee: ast.expr, flags: dict[int, bool], *, strict_attr: bool = False
    ) -> bool:
        """Whether a call from ``caller`` resolves to a flagged helper."""
        name = _qualified_name(callee)
        if not name:
            return False
        bare = name.split(".")[-1]
        candidates = self._helpers.get(bare, [])
        if isinstance(callee, ast.Attribute):
            base = callee.value
            if isinstance(base, ast.Name) and base.id in ("self", "cls"):
                owner = self._owners[id(caller)]
                if owner is None:
                    return False
                lineage = self._lineage(owner)
                return any(
                    flags.get(id(node), False)
                    for node, _binding, method_class in candidates
                    if method_class is not None and id(method_class) in lineage
                )
            module_flags = [
                flags.get(id(node), False) for node, binding, _method_class in candidates if binding is None
            ]
            if strict_attr:
                return bool(module_flags) and all(module_flags)
            return any(module_flags)
        for level in (*reversed(self._chains[id(caller)]), None):
            level_id = None if level is None else id(level)
            bound = [node for node, binding, method_class in candidates if binding == level_id and method_class is None]
            if bound:
                return any(flags.get(id(node), False) for node in bound)
            if bare in self._opaque_names(level):
                return False
        return False


def _resolver(
    index: _ModuleScopes, caller: ast.AST, flags: dict[int, bool], *, strict_attr: bool = False
) -> Callable[[ast.expr], bool]:
    def matches(callee: ast.expr) -> bool:
        return index.resolves_flagged(caller, callee, flags, strict_attr=strict_attr)

    return matches


def _matches_no_helper(_callee: ast.expr) -> bool:
    """Null matcher for the one-hop base predicates."""
    return False


def _assert_no_hand_rolled_exchange_walkers(sources: Mapping[Path, str]) -> None:
    """Transcript walkers must read exchange structure from the shared grammar.

    A function is flagged when a loop reads message role off the items it
    walks AND classifies tool call/result contents (directly or through a
    local classifier helper), and the function carries pairing/boundary state
    — unless it reads the grammar itself (iter_exchanges/pair_results) or
    holds a justified allowlist entry. History-marker handling is exactly what
    hand-rolled walkers forget, and no mechanical rule can require what is
    absent — so the guard keys on the walking constructs themselves.
    Every scope (functions, methods, lambdas) is judged exactly once under
    its qualified name, and helper propagation resolves calls with lexical
    fidelity (nearest binding for bare names, own class for self/cls).
    """
    violations: list[str] = []
    for path, source in sources.items():
        if path == _EXCHANGE_GRAMMAR_MODULE:
            continue
        tree = _tree(path, source)
        index = _ModuleScopes(tree)
        role_flags = {id(scope): bool(_role_read_bases(scope)) for scope in index.scopes}
        id_flags = {id(scope): _reads_tool_id(scope) for scope in index.scopes}
        classifier_flags = {id(scope): _classification_references(scope, _matches_no_helper) for scope in index.scopes}
        grammar_flags = {id(scope): _references_exchange_grammar(scope, _matches_no_helper) for scope in index.scopes}
        for func in index.scopes:
            name = index.qualified[id(func)]
            if (path, name) in _EXCHANGE_WALKER_ALLOWLIST or _references_exchange_grammar(
                func, _resolver(index, func, grammar_flags, strict_attr=True)
            ):
                continue
            walked_names = _walked_item_names(func)
            matches_role_helper = _resolver(index, func, role_flags)
            matches_classifier = _resolver(index, func, classifier_flags)
            flagged = any(
                _reads_walked_role(loop, walked_names, matches_role_helper)
                and _classification_references(loop, matches_classifier)
                for loop in _scoped_walk(func)
                if isinstance(loop, _LOOP_NODES)
            )
            if flagged and _has_pairing_construct(func, _resolver(index, func, id_flags)):
                violations.append(
                    f"{path}:{func.lineno}: {name} hand-rolls an exchange walk; read boundaries and "
                    "pairing from chrys.kernel.exchanges (iter_exchanges/pair_results) or add a justified "
                    "_EXCHANGE_WALKER_ALLOWLIST entry"
                )
    assert violations == [], "\n".join(violations)


def _assert_result_only_classifier_imports_are_allowlisted(sources: Mapping[Path, str]) -> None:
    """Importing the role-gated result-only classifier is a walker smell.

    Consumers get result-only handling from iter_exchanges output; the module
    backstop catches a walker whose loop shape evades the mechanical trigger.
    """
    violations: list[str] = []
    for path, source in sources.items():
        if path == _EXCHANGE_GRAMMAR_MODULE or path in _CLASSIFIER_IMPORT_ALLOWLIST:
            continue
        for node in ast.walk(_tree(path, source)):
            named_import = isinstance(node, ast.ImportFrom) and any(
                alias.name == _RESULT_ONLY_CLASSIFIER for alias in node.names
            )
            # Module-alias access (``ex.is_result_only_message``) reaches the
            # classifier without any ImportFrom, so the use site counts too.
            aliased_use = isinstance(node, ast.Attribute) and node.attr == _RESULT_ONLY_CLASSIFIER
            if named_import or aliased_use:
                violations.append(
                    f"{path}:{node.lineno}: reaching {_RESULT_ONLY_CLASSIFIER} outside the grammar module "
                    "requires a justified _CLASSIFIER_IMPORT_ALLOWLIST entry; consume iter_exchanges instead"
                )
    assert violations == [], "\n".join(violations)


# --- Global TUI locale-context propagation guard ---------------------------

_TUI_ROOT = Path("src/chrys/app/tui")


def _locale_aware_tui_class_index(
    sources: Mapping[Path, str],
) -> tuple[dict[str, list[tuple[Path, int]]], list[str]]:
    """Index locale consumers and return fail-closed inheritance problems."""
    definitions: dict[str, list[tuple[Path, ast.ClassDef]]] = {}
    aliases_by_path: dict[Path, dict[str, str]] = {}
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        tree = _tree(path, source)
        definitions_in_file = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        for node in definitions_in_file:
            definitions.setdefault(node.name, []).append((path, node))
        aliases_by_path[path] = {}

    all_class_names = set(definitions)
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        tree = _tree(path, source)
        aliases_by_path[path].update(
            {
                alias.asname or alias.name: alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
                if alias.name in all_class_names
            }
        )

    locale_aware: set[str] = set()
    for name, class_definitions in definitions.items():
        for _path, node in class_definitions:
            constructor = _class_constructor(node)
            if constructor is not None and _constructor_accepts_locale_controller(constructor):
                locale_aware.add(name)

    ambiguous_inheritance: list[str] = []
    changed = True
    while changed:
        changed = False
        for name, class_definitions in definitions.items():
            if name in locale_aware:
                continue
            inherited_definitions: list[tuple[Path, ast.ClassDef]] = []
            for path, node in class_definitions:
                if _class_constructor(node) is not None:
                    continue
                base_names = {
                    base_name
                    for base in node.bases
                    if (base_name := _local_base_class_name(base, aliases_by_path[path])) is not None
                }
                if base_names.intersection(locale_aware):
                    inherited_definitions.append((path, node))
            if not inherited_definitions:
                continue
            if len(class_definitions) != 1:
                sites = ", ".join(f"{path}:{node.lineno}" for path, node in class_definitions)
                ambiguous_inheritance.append(
                    f"{name} has duplicate definitions and inherits locale context ({sites}); "
                    "class-name graph is ambiguous"
                )
                continue
            path, node = inherited_definitions[0]
            if len(node.bases) != 1:
                ambiguous_inheritance.append(
                    f"{path}:{node.lineno}: {name} inherits locale context through multiple bases; "
                    "define an explicit __init__"
                )
                continue
            locale_aware.add(name)
            changed = True

    sites = {name: [(path, node.lineno) for path, node in definitions[name]] for name in locale_aware}
    return sites, sorted(set(ambiguous_inheritance))


def _locale_aware_tui_class_sites(sources: Mapping[Path, str]) -> dict[str, list[tuple[Path, int]]]:
    """Index explicit and inherited TUI consumers of locale context."""
    sites, problems = _locale_aware_tui_class_index(sources)
    if problems:
        raise AssertionError("\n".join(problems))
    return sites


def _class_constructor(node: ast.ClassDef) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next(
        (
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__"
        ),
        None,
    )


def _constructor_accepts_locale_controller(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    return any(parameter.arg == "locale_controller" for parameter in parameters)


def _local_base_class_name(node: ast.expr, aliases: Mapping[str, str]) -> str | None:
    while isinstance(node, ast.Subscript):
        node = node.value
    qualified = _qualified_name(node)
    if not qualified:
        return None
    bare_name = qualified.rsplit(".", maxsplit=1)[-1]
    return aliases.get(bare_name, bare_name)


def _tui_class_sites(sources: Mapping[Path, str]) -> dict[str, list[tuple[Path, int]]]:
    """Index every TUI class name so bare-name matching can fail closed."""
    sites: dict[str, list[tuple[Path, int]]] = {}
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        for node in ast.walk(_tree(path, source)):
            if isinstance(node, ast.ClassDef):
                sites.setdefault(node.name, []).append((path, node.lineno))
    return sites


def _locale_aware_constructor_references(tree: ast.Module, class_names: set[str]) -> set[str]:
    """Return bare and imported aliases for locale-aware constructor names."""
    references = set(class_names)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        references.update(alias.asname or alias.name for alias in node.names if alias.name in class_names)
    return references


def _assert_tui_locale_controller_propagation_is_explicit(sources: Mapping[Path, str]) -> None:
    """Require explicit locale-context propagation at every TUI-internal constructor edge.

    This is deliberately a structural rule: it requires the
    ``locale_controller=`` keyword and rejects a literal ``None`` on any TUI
    call, but it does not claim that a dynamic expression such as
    ``getattr(...)`` is non-null at runtime. Discovery and call scanning both
    stop at ``_TUI_ROOT``; callers outside that layer are out of scope. Bare
    class-name matching catches package imports without interpreting their
    ``__init__`` re-export chain; class-name ambiguity fails closed instead of
    guessing.
    """
    class_sites, violations = _locale_aware_tui_class_index(sources)
    all_class_sites = _tui_class_sites(sources)
    conflicts = {name: all_class_sites[name] for name in class_sites if len(all_class_sites[name]) != 1}
    if conflicts:
        details = [
            f"{name}: " + ", ".join(f"{path}:{line}" for path, line in sites)
            for name, sites in sorted(conflicts.items())
        ]
        violations.append(
            "locale-aware TUI class names must be unique because the propagation guard "
            "matches bare names across package re-exports:\n" + "\n".join(details)
        )

    class_names = set(class_sites)
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        tree = _tree(path, source)
        references = _locale_aware_constructor_references(tree, class_names)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _qualified_name(node.func)
            keyword = next((item for item in node.keywords if item.arg == "locale_controller"), None)
            if keyword is not None and isinstance(keyword.value, ast.Constant) and keyword.value.value is None:
                violations.append(f"{path}:{node.lineno}: {callee}(...) must not pass literal locale_controller=None")
                continue
            bare_name = callee.rsplit(".", maxsplit=1)[-1]
            if callee not in references and bare_name not in class_names:
                continue
            if keyword is None:
                violations.append(f"{path}:{node.lineno}: {callee}(...) must forward locale_controller= explicitly")
    assert violations == [], "\n".join(violations)


# --- TUI binding-display guard (scans src/chrys, not tests) ----------------

_TUI_BINDING_DISPLAY_MODULE = _TUI_ROOT / "binding_display.py"
_TUI_BINDING_CONSTRUCTION_ALLOWLIST = {
    (
        Path("src/chrys/app/tui/screens/dialogs/agent_load.py"),
        "AgentLoadDialog",
        "Binding",
        "escape",
        "dismiss_if_allowed",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/approval/dialog.py"),
        "ApprovalDialog",
        "Binding",
        "escape",
        "noop",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/approval/dialog.py"),
        "ApprovalDialog",
        "Binding",
        "left",
        "switch_focus",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/approval/dialog.py"),
        "ApprovalDialog",
        "Binding",
        "n,N",
        "decline",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/approval/dialog.py"),
        "ApprovalDialog",
        "Binding",
        "right",
        "switch_focus",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/approval/dialog.py"),
        "ApprovalDialog",
        "Binding",
        "y,Y",
        "approve",
        None,
        False,
    ),
    (Path("src/chrys/app/tui/screens/dialogs/ask_user.py"), "AskUserDialog", "Binding", "escape", "noop", None, False),
    (
        Path("src/chrys/app/tui/screens/dialogs/confirm.py"),
        "ConfirmDialog",
        "Binding",
        "left",
        "switch_focus",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/confirm.py"),
        "ConfirmDialog",
        "Binding",
        "right",
        "switch_focus",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/connection_test.py"),
        "ConnectionTestDialog",
        "Binding",
        "escape",
        "dismiss_if_allowed",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/fork_session.py"),
        "ForkSessionDialog",
        "Binding",
        "escape",
        "dismiss_after_result",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/fork_session.py"),
        "ForkSessionDialog",
        "Binding",
        "left",
        "focus_previous",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/fork_session.py"),
        "ForkSessionDialog",
        "Binding",
        "right",
        "focus_next",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/image_compression.py"),
        "ImageCompressionDialog",
        "Binding",
        "escape",
        "ignore_escape",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/vision_unsupported.py"),
        "VisionUnsupportedDialog",
        "Binding",
        "escape",
        "dismiss_dialog",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/vision_unsupported.py"),
        "VisionUnsupportedDialog",
        "Binding",
        "left",
        "switch_focus",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/dialogs/vision_unsupported.py"),
        "VisionUnsupportedDialog",
        "Binding",
        "right",
        "switch_focus",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/diff/rollback_modal.py"),
        "RollbackProgressModal",
        "Binding",
        "escape",
        "noop",
        None,
        False,
    ),
    (
        Path("src/chrys/app/tui/screens/main/screen.py"),
        "MainScreen",
        "Binding",
        "ctrl+r",
        "prompt_history",
        None,
        False,
    ),
    (Path("src/chrys/app/tui/screens/main/screen.py"), "MainScreen", "Binding", "escape", "escape", None, False),
}
_TUI_PROSE_SINK_ALLOWLIST: set[tuple[Path, str, str]] = {
    # ThinkingIndicator is dead code — zero construction sites repo-wide (verified 2026-08-09).
    (Path("src/chrys/app/tui/widgets/chat/messages.py"), "Static", "thinking"),
    # API-key format hint — data shape, not prose (settled ruling).
    (Path("src/chrys/app/tui/screens/models/screen.py"), "placeholder", "sk-..."),
}
_RICH_MARKUP_TAG_RE = re.compile(r"\[/?[^\[\]]*\]")
_CONTENT_SUBSTITUTION_RE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")
_ASCII_PROSE_RE = re.compile(r"[A-Za-z]{2,}")


def _binding_owner(node: ast.AST, parents: Mapping[int, ast.AST]) -> str:
    owners: list[str] = []
    parent = parents.get(id(node))
    while parent is not None:
        if isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            owners.append(parent.name)
        parent = parents.get(id(parent))
    return ".".join(reversed(owners)) or "<module>"


def _binding_constructor_references(tree: ast.Module) -> set[str]:
    references = {"Binding"}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module != "textual.binding":
            continue
        references.update(alias.asname or alias.name for alias in node.names if alias.name == "Binding")
    return references


def _literal_string(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _display_field_literal(node: ast.expr) -> str:
    # A non-literal display argument cannot match any allowlist entry, so a
    # site that turns dynamic always re-trips the guard for a human decision.
    literal = _literal_string(node)
    return literal if literal is not None else "<dynamic>"


def _binding_site(
    path: Path,
    owner: str,
    kind: str,
    arguments: list[ast.expr],
    keywords: Sequence[ast.keyword] = (),
) -> tuple[Path, str, str, str | None, str | None, str | None, bool | str]:
    key = _literal_string(arguments[0]) if arguments else None
    action = _literal_string(arguments[1]) if len(arguments) > 1 else None
    # The display-relevant fields are part of the site identity: an
    # allowlisted invisible binding that gains a description or flips
    # ``show`` becomes a NEW site and must be re-justified or migrated.
    description = _display_field_literal(arguments[2]) if len(arguments) > 2 else None
    show: bool | str = True
    if len(arguments) > 3:
        positional_show = arguments[3]
        show = (
            positional_show.value
            if isinstance(positional_show, ast.Constant) and isinstance(positional_show.value, bool)
            else "<dynamic>"
        )
    for item in keywords:
        if item.arg == "description" and description is None:
            description = _display_field_literal(item.value)
        elif item.arg == "show":
            value = item.value
            show = value.value if isinstance(value, ast.Constant) and isinstance(value.value, bool) else "<dynamic>"
    return path, owner, kind, key, action, description, show


def _bindings_assignment_value(node: ast.AST) -> ast.expr | None:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "BINDINGS":
        return node.value
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "BINDINGS" for target in node.targets
    ):
        return node.value
    return None


def _has_display_bind_description(node: ast.Call) -> bool:
    positional = node.args[2] if len(node.args) > 2 else None
    keyword = next((item.value for item in node.keywords if item.arg == "description"), None)
    candidates = [candidate for candidate in (positional, keyword) if candidate is not None]
    return any(not (isinstance(candidate, ast.Constant) and candidate.value == "") for candidate in candidates)


def _assert_tui_binding_display_construction_is_canonical(sources: Mapping[Path, str]) -> None:
    """Keep displayed Textual bindings on the registry-populating helper."""
    violations: list[str] = []
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        tree = _tree(path, source)
        parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        binding_references = _binding_constructor_references(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                _qualified_name(node.func) in binding_references
                or _qualified_name(node.func).rsplit(".", maxsplit=1)[-1] == "Binding"
            ):
                owner = _binding_owner(node, parents)
                if path == _TUI_BINDING_DISPLAY_MODULE and owner == "localized_binding":
                    continue
                site = _binding_site(path, owner, "Binding", node.args, node.keywords)
                if site not in _TUI_BINDING_CONSTRUCTION_ALLOWLIST:
                    violations.append(
                        f"{path}:{node.lineno}: construct displayed bindings with localized_binding; "
                        "direct Binding(...) requires a site-specific invisible-binding allowlist entry"
                    )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "bind"
                and _has_display_bind_description(node)
            ):
                violations.append(
                    f"{path}:{node.lineno}: App.bind/BindingsMap.bind display descriptions must use localized_binding"
                )

            value = _bindings_assignment_value(node)
            if not isinstance(value, (ast.List, ast.Tuple)):
                continue
            owner = _binding_owner(node, parents)
            for item in value.elts:
                if not isinstance(item, ast.Tuple) or len(item.elts) not in {2, 3}:
                    continue
                site = _binding_site(path, owner, "tuple", item.elts)
                if site not in _TUI_BINDING_CONSTRUCTION_ALLOWLIST:
                    violations.append(
                        f"{path}:{item.lineno}: BINDINGS tuple shorthand must use localized_binding; "
                        "undisplayed tuples require a site-specific allowlist entry"
                    )
    assert violations == [], "\n".join(violations)


def _tui_callee_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _tui_literal_text(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            fragment.value
            for fragment in node.values
            if isinstance(fragment, ast.Constant) and isinstance(fragment.value, str)
        )
    return None


def _is_tui_prose_bearing_literal(node: ast.expr, *, strip_substitutions: bool = False) -> bool:
    literal = _tui_literal_text(node)
    if literal is None:
        return False
    # Requiring two consecutive ASCII letters after stripping Rich tags lets
    # glyph-only/empty strings, f-string unit fragments ("0s"/"s)"), and
    # markup-tag letters pass without hiding prose from this file-local
    # guard. Only ``from_markup`` templates additionally strip ``$name``
    # substitution tokens (so prose-free skeletons like "[b]$label[/b]"
    # pass): everywhere else ``$name`` renders literally, so it IS the
    # visible prose.
    visible_literal = _RICH_MARKUP_TAG_RE.sub("", literal)
    if strip_substitutions:
        visible_literal = _CONTENT_SUBSTITUTION_RE.sub("", visible_literal)
    return _ASCII_PROSE_RE.search(visible_literal) is not None


def _tui_prose_allowlist_entry(path: Path, sink_tag: str, value: ast.expr) -> tuple[Path, str, str] | None:
    literal = _literal_string(value)
    if literal is None:
        return None
    entry = (path, sink_tag, literal)
    return entry if entry in _TUI_PROSE_SINK_ALLOWLIST else None


def _tui_notify_prose_sites(tree: ast.Module) -> Iterator[tuple[ast.AST, str, str, ast.expr]]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _tui_callee_name(node) != "notify":
            continue
        if node.args:
            yield node, "notify", "message", node.args[0]
        for keyword in node.keywords:
            if keyword.arg in {"message", "title"}:
                yield node, "notify", keyword.arg, keyword.value


def _tui_border_title_prose_sites(tree: ast.Module) -> Iterator[tuple[ast.AST, str, str, ast.expr]]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr in {"border_title", "border_subtitle"}:
                yield node, target.attr, "assignment", node.value


_TUI_WIDGET_LABEL_CALLEES = {"Label", "Button", "Checkbox", "TabPane", "Static"}


def _tui_widget_label_prose_sites(tree: ast.Module) -> Iterator[tuple[ast.AST, str, str, ast.expr]]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        callee_name = _tui_callee_name(node)
        if callee_name in _TUI_WIDGET_LABEL_CALLEES:
            yield node, callee_name, "first positional argument", node.args[0]


def _tui_placeholder_tooltip_prose_sites(tree: ast.Module) -> Iterator[tuple[ast.AST, str, str, ast.expr]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in {"placeholder", "tooltip"}:
                    yield node, keyword.arg, "keyword argument", keyword.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in {"placeholder", "tooltip"}:
                    yield node, target.attr, "assignment", node.value


def _tui_content_markup_prose_sites(tree: ast.Module) -> Iterator[tuple[ast.AST, str, str, ast.expr]]:
    # Localized text belongs in ``$name`` substitutions (kept literal by
    # Content/Text); the markup template itself must stay prose-free. The
    # template parameter is named ``markup`` on Content and ``text`` on
    # rich.Text, so cover both keyword spellings alongside the positional
    # form.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _tui_callee_name(node) != "from_markup":
            continue
        if node.args:
            yield node, "from_markup", "first positional argument", node.args[0]
        for keyword in node.keywords:
            if keyword.arg in {"markup", "text"}:
                yield node, "from_markup", keyword.arg, keyword.value


_TUI_PROSE_SITE_FINDERS = (
    _tui_notify_prose_sites,
    _tui_border_title_prose_sites,
    _tui_widget_label_prose_sites,
    _tui_placeholder_tooltip_prose_sites,
    _tui_content_markup_prose_sites,
)


def _assert_tui_notify_prose_is_localized(sources: Mapping[Path, str]) -> None:
    violations: list[str] = []
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        for node, sink_tag, field, value in _tui_notify_prose_sites(_tree(path, source)):
            if _is_tui_prose_bearing_literal(value) and _tui_prose_allowlist_entry(path, sink_tag, value) is None:
                violations.append(f"{path}:{node.lineno}: localize raw prose in notify {field}")
    assert violations == [], "\n".join(violations)


def _assert_tui_border_titles_are_localized(sources: Mapping[Path, str]) -> None:
    violations: list[str] = []
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        for node, sink_tag, _field, value in _tui_border_title_prose_sites(_tree(path, source)):
            if _is_tui_prose_bearing_literal(value) and _tui_prose_allowlist_entry(path, sink_tag, value) is None:
                violations.append(f"{path}:{node.lineno}: localize raw prose assigned to {sink_tag}")
    assert violations == [], "\n".join(violations)


def _assert_tui_widget_label_prose_is_localized(sources: Mapping[Path, str]) -> None:
    violations: list[str] = []
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        for node, sink_tag, field, value in _tui_widget_label_prose_sites(_tree(path, source)):
            if _is_tui_prose_bearing_literal(value) and _tui_prose_allowlist_entry(path, sink_tag, value) is None:
                violations.append(f"{path}:{node.lineno}: localize raw prose in {sink_tag} {field}")
    assert violations == [], "\n".join(violations)


def _assert_tui_placeholder_tooltip_prose_is_localized(sources: Mapping[Path, str]) -> None:
    violations: list[str] = []
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        for node, sink_tag, field, value in _tui_placeholder_tooltip_prose_sites(_tree(path, source)):
            if _is_tui_prose_bearing_literal(value) and _tui_prose_allowlist_entry(path, sink_tag, value) is None:
                violations.append(f"{path}:{node.lineno}: localize raw prose in {sink_tag} {field}")
    assert violations == [], "\n".join(violations)


def _assert_tui_content_markup_prose_is_localized(sources: Mapping[Path, str]) -> None:
    violations: list[str] = []
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        for node, sink_tag, field, value in _tui_content_markup_prose_sites(_tree(path, source)):
            if (
                _is_tui_prose_bearing_literal(value, strip_substitutions=True)
                and _tui_prose_allowlist_entry(path, sink_tag, value) is None
            ):
                violations.append(f"{path}:{node.lineno}: localize raw prose in {sink_tag} {field}")
    assert violations == [], "\n".join(violations)


def _textual_content_references(tree: ast.Module) -> set[str]:
    """Return import spellings that name :class:`textual.content.Content`."""
    references: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "textual.content":
            references.update(alias.asname or alias.name for alias in node.names if alias.name == "Content")
        elif isinstance(node, ast.ImportFrom) and node.module == "textual":
            references.update(
                f"{alias.asname or alias.name}.Content" for alias in node.names if alias.name == "content"
            )
        elif isinstance(node, ast.Import):
            references.update(
                f"{alias.asname}.Content" if alias.asname else "textual.content.Content"
                for alias in node.names
                if alias.name == "textual.content"
            )
    return references


def _assert_tui_content_from_text_disables_markup(sources: Mapping[Path, str]) -> None:
    """Keep dynamic display strings out of Textual's implicit markup parser."""
    violations: list[str] = []
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        tree = _tree(path, source)
        callees = {f"{reference}.from_text" for reference in _textual_content_references(tree)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _qualified_name(node.func) not in callees:
                continue
            markup = next((keyword.value for keyword in node.keywords if keyword.arg == "markup"), None)
            if not (isinstance(markup, ast.Constant) and markup.value is False):
                violations.append(
                    f"{path}:{node.lineno}: Textual Content.from_text must pass markup=False; "
                    "use Content.from_markup for intentional markup"
                )
    assert violations == [], "\n".join(violations)


_I18N_MESSAGES_PATH = Path("src/chrys/foundation/i18n/messages.py")
_I18N_INIT_PATH = Path("src/chrys/foundation/i18n/__init__.py")


def _assert_i18n_message_construction_is_canonical(sources: Mapping[Path, str]) -> None:
    """Keep every extractable message on the one AST-visible construction path."""
    violations: list[str] = []
    for path, source in sources.items():
        if path == _I18N_MESSAGES_PATH:
            continue
        tree = _tree(path, source)
        parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        # Ordinals, not line numbers: semicolon-joined statements share a
        # line while still executing strictly in statement order.
        top_level_ordinal = {
            id(descendant): index for index, statement in enumerate(tree.body) for descendant in ast.walk(statement)
        }
        canonical_import_ordinal: int | None = None
        i18n_msg_import = False
        i18n_module_refs: set[str] = set()
        i18n_module_bindings: set[str] = set()
        local_message_definitions: set[str] = set()
        dataclasses_module_names = {"dataclasses"}
        replace_function_names = {"replace"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # An unaliased plain import binds its ROOT component, so
                    # ``import msg.submodule`` rebinds msg.
                    if (alias.asname or alias.name.split(".", maxsplit=1)[0]) == "msg":
                        violations.append(f"{path}:{node.lineno}: msg may not be locally defined or rebound")
                    elif alias.name == "chrys.foundation.i18n" or alias.name.startswith("chrys.foundation.i18n."):
                        if alias.asname is not None:
                            i18n_module_refs.add(alias.asname)
                            i18n_module_bindings.add(alias.asname)
                        else:
                            # An unaliased deep import makes every prefix
                            # module a live dotted value too.
                            name = alias.name
                            while name.startswith("chrys.foundation.i18n"):
                                i18n_module_refs.add(name)
                                name = name.rsplit(".", maxsplit=1)[0]
                    elif alias.name == "dataclasses" and alias.asname is not None:
                        dataclasses_module_names.add(alias.asname)
            elif isinstance(node, ast.ImportFrom):
                module = _resolved_import_module(path, node)
                i18n_family = module is not None and module.startswith("chrys.foundation.i18n")
                for alias in node.names:
                    # Any import binding the name msg other than an i18n
                    # msg-member import (e.g. MessageDef as msg) is a rebind,
                    # whatever the source module.
                    if (alias.asname or alias.name) == "msg" and not (i18n_family and alias.name in {"msg", "*"}):
                        violations.append(f"{path}:{node.lineno}: msg may not be locally defined or rebound")
                        continue
                    if module == "chrys.foundation":
                        if alias.name == "i18n":
                            i18n_module_refs.add(alias.asname or alias.name)
                            i18n_module_bindings.add(alias.asname or alias.name)
                    elif module == "dataclasses":
                        if alias.name == "replace":
                            replace_function_names.add(alias.asname or alias.name)
                    elif i18n_family:
                        if alias.name not in {"msg", "*"}:
                            # Submodule and member aliases become owner refs so
                            # qualified constructor calls through them (e.g.
                            # messages.msg) are caught below.
                            i18n_module_refs.add(alias.asname or alias.name)
                            continue
                        package_reexport = (
                            path == _I18N_INIT_PATH and node.level == 0 and module == "chrys.foundation.i18n.messages"
                        )
                        # Only an unconditional top-level import is canonical:
                        # a conditional one can lose to a rogue same-name
                        # binding at runtime.
                        canonical = (
                            node.level == 0
                            and module == "chrys.foundation.i18n"
                            and alias.asname is None
                            and node in tree.body
                        )
                        if package_reexport and alias.asname is None:
                            continue
                        i18n_msg_import = True
                        if canonical and alias.name == "msg":
                            ordinal = top_level_ordinal[id(node)]
                            if canonical_import_ordinal is None or ordinal < canonical_import_ordinal:
                                canonical_import_ordinal = ordinal
                        else:
                            violations.append(
                                f"{path}:{node.lineno}: import msg only with 'from chrys.foundation.i18n import msg'"
                            )

        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "msg"
                and len(targets) == 1
                and isinstance(targets[0], ast.Name)
            ):
                local_message_definitions.add(targets[0].id)

        for node in ast.walk(tree):
            if (
                (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.MatchAs, ast.MatchStar))
                    and node.name == "msg"
                )
                or (isinstance(node, ast.MatchMapping) and node.rest == "msg")
                or (
                    # ExceptHandler.name is a plain string, invisible to Name
                    # Store/Del checks.
                    isinstance(node, ast.ExceptHandler) and node.name == "msg"
                )
            ):
                violations.append(f"{path}:{node.lineno}: msg may not be locally defined or rebound")
            if i18n_msg_import and isinstance(node, ast.arg) and node.arg == "msg":
                violations.append(f"{path}:{node.lineno}: msg may not be passed or rebound as a local name")
            if isinstance(node, ast.Name) and node.id in i18n_module_bindings:
                # Module bindings may only serve qualified attribute access;
                # aliasing one to a new name would launder later .msg calls.
                parent = parents.get(id(node))
                if not (isinstance(parent, ast.Attribute) and parent.value is node):
                    violations.append(
                        f"{path}:{node.lineno}: i18n module references may not be aliased or passed as values"
                    )
            if isinstance(node, ast.Attribute):
                # The dotted spelling of an unaliased plain import launders
                # the same way a name binding would; deep imports record every
                # i18n-namespace prefix as a ref, while member tails
                # (….messages.MessageDef) stay legal for annotations.
                qualified_ref = _qualified_name(node)
                if "." in qualified_ref and qualified_ref in i18n_module_refs:
                    parent = parents.get(id(node))
                    if not (isinstance(parent, ast.Attribute) and parent.value is node):
                        violations.append(
                            f"{path}:{node.lineno}: i18n module references may not be aliased or passed as values"
                        )

            if isinstance(node, ast.Attribute) and node.attr == "msg":
                # Any access — not just a direct call — or the attribute can
                # be aliased to a local factory and called untracked.
                owner = _qualified_name(node.value)
                if _is_i18n_owner(owner, i18n_module_refs):
                    violations.append(
                        f"{path}:{node.lineno}: call msg as a bare name, never through a module or object"
                    )

            if not isinstance(node, ast.Call):
                continue
            qualified = _qualified_name(node.func)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "msg"
                and (
                    canonical_import_ordinal is None
                    # A call in a statement before the import raises NameError
                    # at runtime while extraction would still record it.
                    or top_level_ordinal.get(id(node), -1) < canonical_import_ordinal
                    or not _is_module_level_message_assignment(node, parents, tree)
                )
            ):
                violations.append(
                    f"{path}:{node.lineno}: msg() must follow the canonical import and be a direct "
                    "module-level assignment value"
                )
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in {"msg", "MessageDef", "MessageRef"}
            ):
                owner = _qualified_name(node.args[0])
                if _is_i18n_owner(owner, i18n_module_refs):
                    violations.append(
                        f"{path}:{node.lineno}: access message constructors as imported names, never via getattr"
                    )
            if qualified.rsplit(".", maxsplit=1)[-1] in {"MessageDef", "MessageRef"}:
                violations.append(f"{path}:{node.lineno}: construct messages only through msg() and MessageDef.bind()")
            is_replace_call = (isinstance(node.func, ast.Name) and node.func.id in replace_function_names) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace"
                and _qualified_name(node.func.value) in dataclasses_module_names
            )
            if is_replace_call and any(
                isinstance(descendant, (ast.Name, ast.Attribute))
                and (
                    _qualified_name(descendant).rsplit(".", maxsplit=1)[-1] in {"MessageDef", "MessageRef"}
                    or (isinstance(descendant, ast.Name) and descendant.id in local_message_definitions)
                )
                for descendant in ast.walk(node)
            ):
                violations.append(
                    f"{path}:{node.lineno}: dataclasses.replace must not construct MessageDef or MessageRef variants"
                )

        for node in ast.walk(tree):
            if not i18n_msg_import or not isinstance(node, ast.Name) or node.id != "msg":
                continue
            parent = parents.get(id(node))
            if isinstance(parent, ast.Call) and parent.func is node:
                continue
            violations.append(f"{path}:{node.lineno}: msg is a construction function, not a first-class value")
    assert violations == [], "\n".join(violations)


def _is_i18n_owner(owner: str, module_refs: set[str]) -> bool:
    # An owner reaches the i18n namespace when any dotted prefix of it is a
    # recorded module ref (i18n.messages through `from chrys.foundation
    # import i18n`) or any segment is spelled i18n — nested submodule access
    # launders the same way single-step access does.
    parts = owner.split(".")
    if "i18n" in parts:
        return True
    prefix = parts[0]
    if prefix in module_refs:
        return True
    for part in parts[1:]:
        prefix = f"{prefix}.{part}"
        if prefix in module_refs:
            return True
    return False


def _resolved_import_module(path: Path, node: ast.ImportFrom) -> str | None:
    """Resolve a possibly-relative import to its absolute module name."""
    if node.level == 0:
        return node.module
    parts = list(path.with_suffix("").parts)
    if "src" in parts:
        parts = parts[parts.index("src") + 1 :]
    package = parts[:-1]
    ascent = node.level - 1
    if ascent > len(package):
        return None
    base = package[: len(package) - ascent] if ascent else package
    prefix = ".".join(base)
    if node.module:
        return f"{prefix}.{node.module}" if prefix else node.module
    return prefix or None


def _is_module_level_message_assignment(
    call: ast.Call,
    parents: Mapping[int, ast.AST],
    tree: ast.Module,
) -> bool:
    parent = parents.get(id(call))
    if isinstance(parent, ast.Assign):
        return (
            parent.value is call
            and len(parent.targets) == 1
            and isinstance(parent.targets[0], ast.Name)
            and parent in tree.body
        )
    if isinstance(parent, ast.AnnAssign):
        return parent.value is call and isinstance(parent.target, ast.Name) and parent in tree.body
    return False


_HYGIENE_RULES = (
    _assert_no_scroll_relative,
    _assert_no_unapproved_local_polling_helpers,
    _assert_no_ignored_wait_until_results,
    _assert_integration_marker_directory_disjoint,
    _assert_no_direct_agent_engine_start,
    _assert_quarantine_marker_metadata,
    _assert_tool_loop_layer_uses_are_invariant_checked,
    _assert_trajectory_prefix_checks_use_physical_slots,
)

_SRC_HYGIENE_RULES = (
    _assert_subprocess_stdin_is_explicit,
    _assert_optional_extra_imports_are_function_scoped,
    _assert_no_hand_rolled_exchange_walkers,
    _assert_result_only_classifier_imports_are_allowlisted,
    _assert_tui_binding_display_construction_is_canonical,
    _assert_tui_notify_prose_is_localized,
    _assert_tui_border_titles_are_localized,
    _assert_tui_widget_label_prose_is_localized,
    _assert_tui_placeholder_tooltip_prose_is_localized,
    _assert_tui_content_markup_prose_is_localized,
    _assert_tui_content_from_text_disables_markup,
    _assert_i18n_message_construction_is_canonical,
)

_GLOBAL_SRC_HYGIENE_RULES = (_assert_tui_locale_controller_propagation_is_explicit,)


@pytest.mark.parametrize("shard", range(_SWEEP_SHARDS))
def test_hygiene_rules_hold_across_test_sources(shard: int) -> None:
    """Run every hygiene rule against ONE shared read+parse per test file.

    Deliberately NOT one test per rule: that shape re-read and re-parsed all
    ~330 test files PER RULE (9-15s each on contended CI macOS runners), and
    xdist may schedule the tests onto different workers, so caching alone
    cannot share the work. Within a shard each file is read and parsed once
    and every rule runs against that single cached parse — rules are all
    file-local (no cross-file state), which is also what makes sharding sound:
    each file is checked exactly once, in exactly one shard.

    The tree is released after each file: holding all ~330 ASTs concurrently
    spiked worker memory by hundreds of MB, enough to crash an xdist worker on
    memory-constrained macOS CI runners. And the sweep is sharded because the
    monolithic version measured 53-55s on contended CI draws — 90% of the
    global 60s per-test timeout, and the #1 tail pole on every platform.
    Failures are aggregated so one run still reports every rule's violations.
    """
    failures: list[str] = []
    for path, source in _test_sources(shard).items():
        single_source = {path: source}
        for rule in _HYGIENE_RULES:
            try:
                rule(single_source)
            except AssertionError as exc:
                failures.append(str(exc))
        _tree.cache_clear()
    assert not failures, "\n\n".join(failures)


@pytest.mark.parametrize("shard", range(_SWEEP_SHARDS))
def test_exchange_walker_guard_holds_across_src_sources(shard: int) -> None:
    """Run the src-scoped rules against ONE shared read+parse per src file.

    Same sharded single-parse shape as the test-source sweep above, for the
    same CI wall-clock and worker-memory reasons.
    """
    failures: list[str] = []
    for path, source in _src_sources(shard).items():
        single_source = {path: source}
        for rule in _SRC_HYGIENE_RULES:
            try:
                rule(single_source)
            except AssertionError as exc:
                failures.append(str(exc))
        _tree.cache_clear()
    assert not failures, "\n\n".join(failures)


def test_sweep_shard_partitions_are_complete_disjoint_and_non_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sharded sweeps must cover every enumerated path exactly once."""
    _global_definition_lines()
    monkeypatch.setattr(Path, "read_text", lambda _path, *, encoding: "")
    problems: list[str] = []
    for collector, root in ((_test_sources, TESTS_ROOT), (_src_sources, SRC_ROOT / "chrys")):
        enumerated_keys = {path.relative_to(REPO_ROOT) for path in root.rglob("*.py")}
        all_keys = set(collector())
        collector_missing = enumerated_keys - all_keys
        collector_unexpected = all_keys - enumerated_keys
        if collector_missing or collector_unexpected:
            details = []
            if collector_missing:
                details.append(
                    f"omitted from shard=None: {', '.join(path.as_posix() for path in sorted(collector_missing))}"
                )
            if collector_unexpected:
                details.append(
                    "not present in the filesystem enumeration: "
                    f"{', '.join(path.as_posix() for path in sorted(collector_unexpected))}"
                )
            problems.append(
                _meta_guard_problem(
                    collector.__name__,
                    f"{collector.__name__}(shard=None) differs from the complete filesystem key-set "
                    f"({'; '.join(details)})",
                    "enumerate every *.py file under the collector root without an extra skip or early continue",
                )
            )
        shard_keys = [set(collector(shard)) for shard in range(_SWEEP_SHARDS)]
        union = set().union(*shard_keys)
        missing = all_keys - union
        unexpected = union - all_keys
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing from all shards: {', '.join(path.as_posix() for path in sorted(missing))}")
            if unexpected:
                details.append(
                    f"absent from the unfiltered collector: {', '.join(path.as_posix() for path in sorted(unexpected))}"
                )
            problems.append(
                _meta_guard_problem(
                    collector.__name__,
                    f"{collector.__name__} shard union differs from its shard=None key-set ({'; '.join(details)})",
                    "make _shard_of return only values in range(_SWEEP_SHARDS) and keep the filtered and "
                    "unfiltered collector paths identical",
                )
            )
        for left in range(_SWEEP_SHARDS):
            for right in range(left + 1, _SWEEP_SHARDS):
                overlap = shard_keys[left] & shard_keys[right]
                if overlap:
                    problems.append(
                        _meta_guard_problem(
                            collector.__name__,
                            f"{collector.__name__} shards {left} and {right} overlap at "
                            f"{', '.join(path.as_posix() for path in sorted(overlap))}",
                            "assign every relative path to exactly one shard in _shard_of",
                        )
                    )
        for shard, keys in enumerate(shard_keys):
            if not keys:
                problems.append(
                    _meta_guard_problem(
                        collector.__name__,
                        f"{collector.__name__} shard {shard} is empty",
                        "choose _SWEEP_SHARDS and _shard_of so every parametrized shard receives at least one file",
                    )
                )

    assert problems == [], "\n".join(problems)


def test_global_src_hygiene_rules_hold_across_all_sources() -> None:
    """Run cross-file rules once over the complete source graph."""
    assert _GLOBAL_SRC_HYGIENE_RULES, "global source hygiene registry must not be empty"
    sources = _src_sources()
    failures: list[str] = []
    for rule in _GLOBAL_SRC_HYGIENE_RULES:
        try:
            rule(sources)
        except AssertionError as exc:
            failures.append(str(exc))
    _tree.cache_clear()
    assert not failures, "\n\n".join(failures)


def _function_import_bindings(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    bindings: dict[str, str] = {}

    class ImportVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
            if child is node:
                self.generic_visit(child)

        def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
            if child is node:
                self.generic_visit(child)

        def visit_ClassDef(self, child: ast.ClassDef) -> None:
            return

        def visit_ImportFrom(self, child: ast.ImportFrom) -> None:
            if child.module is None:
                return
            for alias in child.names:
                bindings[alias.asname or alias.name] = child.module

    ImportVisitor().visit(node)
    return bindings


def _is_bootstrap_entrypoint_module(module: str) -> bool:
    """Recognise an app-tier module a dispatch helper hands control to.

    Restricting this to ``chrys.app.cli.*`` dropped ``chrys install``, which
    dispatches into chrys.app.installer — a real command that was therefore
    neither checked nor exempted.
    """
    return module.startswith("chrys.app.")


def _entrypoint_dispatch_paths(source: str) -> set[Path]:
    """Resolve the runtime modules ``app.py::main`` dispatches to.

    This reads the two dispatch shapes the CLI actually uses today: ``main``
    returning a call into an imported ``chrys.app.cli.*`` module, and ``main``
    returning a ``_run_*`` helper that calls into an app-tier module. A command
    written some third way would not be discovered here — following arbitrary
    call graphs was weighed and rejected as more machinery than the dispatcher
    warrants while that convention holds. Adding a dispatch shape means
    teaching this function about it.
    """
    tree = ast.parse(source, filename="src/chrys/app/cli/app.py")
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    main = functions.get("main")
    if main is None:
        return set()

    dispatched_modules: set[str] = set()
    helpers: set[str] = set()
    main_bindings = _function_import_bindings(main)
    for node in ast.walk(main):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
            continue
        called = _qualified_name(node.value.func)
        if called.startswith("_run_"):
            helpers.add(called)
            continue
        module = main_bindings.get(called)
        if module is not None and module.startswith("chrys.app.cli."):
            dispatched_modules.add(module)

    for helper_name in helpers:
        helper = functions.get(helper_name)
        if helper is None:
            continue
        bindings = _function_import_bindings(helper)
        for node in ast.walk(helper):
            if not isinstance(node, ast.Call):
                continue
            module = bindings.get(_qualified_name(node.func))
            if module is not None and _is_bootstrap_entrypoint_module(module):
                dispatched_modules.add(module)

    return {Path("src") / Path(*module.split(".")).with_suffix(".py") for module in dispatched_modules}


def _calls_bootstrap_runtime(path: Path, source: str) -> bool:
    return any(
        isinstance(node, ast.Call) and _qualified_name(node.func).rsplit(".", maxsplit=1)[-1] == "bootstrap_runtime"
        for node in ast.walk(_tree(path, source))
    )


def test_entrypoint_bootstrap_completeness_and_exemptions_are_live() -> None:
    """Every CLI-dispatched runtime module must bootstrap or remain explicitly exempt."""
    sources = _src_sources()
    dispatcher_path = Path("src/chrys/app/cli/app.py")
    dispatcher_source = sources.get(dispatcher_path)
    if dispatcher_source is None:
        pytest.fail(
            f"{dispatcher_path}:1: entrypoint-bootstrap-completeness cannot find the dispatcher; violates "
            "AGENTS.md:31 (every entrypoint calls bootstrap_runtime). Fix: restore app/cli/app.py or update this "
            "guard to the real dispatcher"
        )
    dispatched_paths = _entrypoint_dispatch_paths(dispatcher_source)
    problems: list[str] = []

    for path, reason in _ENTRYPOINT_BOOTSTRAP_EXEMPT.items():
        if path not in sources:
            problems.append(
                f"{path}:1: entrypoint bootstrap exemption ({reason}) names no real file; violates AGENTS.md:129 "
                '("A guard that can\'t go red is worse than none — it is believed."). Fix: remove the stale '
                "exemption or point it at the live dispatched module"
            )
        elif path not in dispatched_paths:
            problems.append(
                f"{path}:1: entrypoint bootstrap exemption ({reason}) is not reachable from app.py::main; violates "
                'AGENTS.md:129 ("A guard that can\'t go red is worse than none — it is believed."). Fix: remove '
                "the stale exemption or restore the real dispatch edge"
            )

    for path in sorted(dispatched_paths):
        source = sources.get(path)
        if source is None:
            problems.append(
                f"{path}:1: app.py::main dispatches to a module with no source file; violates AGENTS.md:31 "
                "(every entrypoint calls bootstrap_runtime). Fix: restore the dispatched module or correct the "
                "dispatcher import"
            )
        elif path not in _ENTRYPOINT_BOOTSTRAP_EXEMPT and not _calls_bootstrap_runtime(path, source):
            problems.append(
                f"{path}:1: entrypoint-bootstrap-completeness found no bootstrap_runtime() call; violates "
                "AGENTS.md:31 (every entrypoint calls bootstrap_runtime; never duplicate). Fix: call "
                "bootstrap_runtime through the module's runtime-preparation path, or add a narrowly justified "
                "live exemption"
            )

    _tree.cache_clear()
    assert problems == [], "\n".join(problems)


def test_every_hygiene_rule_is_registered() -> None:
    """A rule that exists but is missing from the registries is silently unenforced."""
    registries = (_HYGIENE_RULES, _SRC_HYGIENE_RULES, _GLOBAL_SRC_HYGIENE_RULES)
    registered = [rule for registry in registries for rule in registry]
    assert len(registered) == len(set(registered)), "hygiene rule registries must be pairwise disjoint"
    assert _assert_tui_locale_controller_propagation_is_explicit in _GLOBAL_SRC_HYGIENE_RULES, (
        "locale-controller propagation needs the complete source graph and must stay in the global registry"
    )
    declared = {obj for name, obj in globals().items() if name.startswith("_assert_") and callable(obj)}
    assert declared == set(registered)


def _observed_public_import_roots() -> dict[str, set[str]]:
    """Group installed import roots by distribution, dropping non-source names."""
    roots: dict[str, set[str]] = {}
    for package, distributions in importlib.metadata.packages_distributions().items():
        if package.startswith("_") or "__mypyc" in package:
            continue
        for distribution in distributions:
            roots.setdefault(_normalized_distribution_name(distribution), set()).add(package)
    return roots


def test_stated_import_roots_match_the_installed_environment() -> None:
    """Catch table drift wherever the environment can still adjudicate it."""
    base, optional = _project_distributions()
    observed = _observed_public_import_roots()
    problems: list[str] = []
    for distribution in sorted(base | optional):
        if distribution not in observed:
            continue
        stated = _import_roots(distribution)
        # The environment only reports top-level roots, so a qualified prefix
        # is checked at the granularity it can actually adjudicate.
        stated_roots = {prefix.split(".", maxsplit=1)[0] for prefix in stated}
        if stated_roots != observed[distribution]:
            problems.append(
                _meta_guard_problem(
                    "_DISTRIBUTION_IMPORT_ROOTS",
                    f"{distribution} supplies {sorted(observed[distribution])} but the table resolves it to "
                    f"{sorted(stated)}",
                    "correct the _DISTRIBUTION_IMPORT_ROOTS entry (or add one) so the stated roots match",
                )
            )

    assert problems == [], "\n".join(problems)


def test_stated_import_roots_have_no_stale_entries() -> None:
    base, optional = _project_distributions()
    declared = base | optional
    problems = [
        _meta_guard_problem(
            "_DISTRIBUTION_IMPORT_ROOTS",
            f"entry {distribution} is not a declared project dependency",
            "remove the stale entry or restore the dependency in pyproject.toml",
        )
        for distribution in sorted(_DISTRIBUTION_IMPORT_ROOTS)
        if distribution not in declared
    ]

    assert problems == [], "\n".join(problems)


def test_optional_roots_survive_an_uninstalled_distribution() -> None:
    """The counterexample the environment-derived mapping used to lose.

    pywinpty is Windows-marked, so it is absent on the job that runs this
    sweep. Its import root must still be covered, or a module-scope
    ``import winpty`` below the app tier would go unnoticed.
    """
    assert "winpty" in _optional_only_import_prefixes()


@_pins_allowlist("_OPTIONAL_IMPORT_ALLOWLIST")
def test_optional_import_allowlist_entries_are_live() -> None:
    """An exempted module must still be the thing the rule would flag."""
    problems: list[str] = []
    for path in sorted(_OPTIONAL_IMPORT_ALLOWLIST):
        absolute = REPO_ROOT / path
        if not absolute.is_file():
            problems.append(
                _meta_guard_problem(
                    "_OPTIONAL_IMPORT_ALLOWLIST",
                    f"entry {path} names a missing file",
                    "remove the stale entry or update it to the live module",
                )
            )
            continue
        collector = _ModuleScopeImportCollector()
        collector.visit(ast.parse(absolute.read_text(encoding="utf-8")))
        if not any(_optional_prefix_for(imported) for imported, _line in collector.imports):
            problems.append(
                _meta_guard_problem(
                    "_OPTIONAL_IMPORT_ALLOWLIST",
                    f"entry {path} has no module-scope optional-only import, so the rule would not flag it",
                    "remove the stale entry now that the module no longer needs the exemption",
                )
            )

    assert problems == [], "\n".join(problems)


def test_optional_import_allowlist_modules_stay_lazily_reached() -> None:
    """The exemption's premise: nothing pulls these in at import time.

    A module-scope import of an exempted module anywhere in the tree would
    carry the optional dependency straight back onto the bootstrap path, which
    is the crash the rule exists to prevent.
    """
    exempted = {".".join(path.relative_to(Path("src")).with_suffix("").parts) for path in _OPTIONAL_IMPORT_ALLOWLIST}
    violations: list[str] = []
    for path in sorted((REPO_ROOT / "src" / "chrys").rglob("*.py")):
        collector = _ResolvedModuleScopeImportCollector(path)
        collector.visit(ast.parse(path.read_text(encoding="utf-8")))
        relative = path.relative_to(REPO_ROOT)
        violations.extend(
            f"{relative}:{line}: {imported!r} is exempted from optional-extra-module-scope-imports only because "
            "nothing imports it at import time; violates AGENTS.md:3 (pyproject.toml is the source of truth for "
            "deps). Fix: move this import inside the function that gates the optional dependency"
            for imported, line in collector.imports
            if imported in exempted and relative not in _OPTIONAL_IMPORT_ALLOWLIST
        )

    assert violations == [], "\n".join(violations)


@pytest.mark.parametrize(
    "source",
    [
        "from chrys.foundation.observability import exporters\n",
        "from . import exporters\n",
    ],
    ids=["package-form", "relative-package-form"],
)
def test_optional_import_allowlist_lazy_pin_resolves_submodule_aliases(source: str) -> None:
    """Package-form imports must not hide an exempted optional module."""
    path = REPO_ROOT / "src/chrys/foundation/observability/consumer.py"
    collector = _ResolvedModuleScopeImportCollector(path)
    collector.visit(ast.parse(source))

    assert ("chrys.foundation.observability.exporters", 1) in collector.imports


@pytest.mark.parametrize(
    "body",
    [
        "    subprocess.run(['probe'], stdin=None)\n",
        "    subprocess.run(['probe'], input=None)\n",
        "    kwargs = {'stdin': None}\n    subprocess.run(['probe'], **kwargs)\n",
    ],
)
def test_subprocess_stdin_guard_rejects_explicit_none(body: str) -> None:
    """``stdin=None`` is the inheriting default spelled out, not a decision."""
    source = "import subprocess\ndef spawn(**kwargs):\n" + body

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    assert "AGENTS.md:39" in str(exc_info.value)


@pytest.mark.parametrize(
    "invalidation",
    [
        '    kwargs["stdin"] = None',
        "    kwargs = {}",
        '    del kwargs["stdin"]',
        '    kwargs.pop("stdin")',
        "    kwargs.clear()",
        '    saved = kwargs.pop("stdin")',
        '    kwargs.update({"stdin": None})',
        '    if cond:\n        kwargs["stdin"] = None',
        '    if cond:\n        del kwargs["stdin"]',
    ],
    ids=[
        "overwrite-none",
        "rebind",
        "delete",
        "pop",
        "clear",
        "pop-as-value",
        "update-none",
        "conditional-overwrite",
        "conditional-delete",
    ],
)
def test_subprocess_stdin_guard_kills_invalidated_evidence(invalidation: str) -> None:
    """Evidence is a *must* judgement; its retraction is a *may* judgement.

    A removal that only sometimes runs still leaves the child able to inherit
    our stdin, so it has to count even from inside a branch.
    """
    source = (
        "import subprocess\n"
        "def spawn(cond, **kwargs):\n"
        '    kwargs["stdin"] = subprocess.DEVNULL\n'
        f"{invalidation}\n"
        "    subprocess.run(['probe'], **kwargs)\n"
    )

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    assert "AGENTS.md:39" in str(exc_info.value)


@pytest.mark.parametrize(
    "statement",
    [
        '    if cond:\n        kwargs["stdin"] = subprocess.DEVNULL',
        '    if cond:\n        kwargs["creationflags"] = 0',
        "    kwargs.update(extra())",
        '    def later():\n        kwargs.pop("stdin")',
        '    later = lambda: kwargs.pop("stdin")',
    ],
    ids=["re-default", "unrelated-key", "opaque-update", "nested-def", "lambda"],
)
def test_subprocess_stdin_guard_keeps_evidence_through_harmless_statements(statement: str) -> None:
    """Only a statically certain retraction counts, or the rule cries wolf.

    Re-defaulting inside a branch must not read as a retraction, and a removal
    parked in a callable that may never run is not one either.
    """
    source = (
        "import subprocess\n"
        "def spawn(cond, extra, **kwargs):\n"
        '    kwargs["stdin"] = subprocess.DEVNULL\n'
        f"{statement}\n"
        "    subprocess.run(['probe'], **kwargs)\n"
    )

    _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})


@pytest.mark.parametrize(
    ("label", "spawn"),
    [
        ("to_thread", "asyncio.to_thread(subprocess.Popen, ['probe'], **kwargs)"),
        ("run_in_executor", "loop.run_in_executor(None, subprocess.Popen, ['probe'], **kwargs)"),
        ("partial", "loop.run_in_executor(None, partial(subprocess.Popen, ['probe'], **kwargs))"),
    ],
)
def test_subprocess_stdin_guard_sees_through_thread_offload(label: str, spawn: str) -> None:
    """Handing the constructor to a thread pool must not hide the spawn.

    The detached hook worker is launched exactly this way, so a callee-name
    test alone would leave the rule blind to its own motivating case.
    """
    source = f"import asyncio\nimport subprocess\nfrom functools import partial\ndef go(loop, **kwargs):\n    {spawn}\n"

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    assert "AGENTS.md:39" in str(exc_info.value), label


@pytest.mark.parametrize(
    ("import_statement", "partial_call"),
    [
        ("from functools import partial as bind", "bind"),
        ("import functools as ft", "ft.partial"),
    ],
    ids=["member-alias", "module-alias"],
)
def test_subprocess_stdin_guard_sees_partial_aliases(import_statement: str, partial_call: str) -> None:
    source = (
        "import subprocess\n"
        f"{import_statement}\n"
        "def go(loop):\n"
        f"    loop.run_in_executor(None, {partial_call}(subprocess.Popen, ['probe']))\n"
    )

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    assert "AGENTS.md:39" in str(exc_info.value)


def test_subprocess_stdin_guard_accepts_offload_with_dominating_default() -> None:
    source = (
        "import asyncio\n"
        "import subprocess\n"
        "def go(**kwargs):\n"
        '    kwargs["stdin"] = subprocess.DEVNULL\n'
        "    try:\n"
        "        return asyncio.to_thread(subprocess.Popen, ['probe'], **kwargs)\n"
        "    finally:\n"
        "        pass\n"
    )

    _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})


def test_subprocess_stdin_guard_rejects_call_result_splat() -> None:
    source = (
        "import subprocess\n"
        "def helper():\n"
        "    return {'stdin': subprocess.DEVNULL}\n"
        "subprocess.run(['probe'], **helper())\n"
    )

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    message = str(exc_info.value)
    assert "src/chrys/foundation/bad_probe.py:4" in message
    assert "AGENTS.md:39" in message
    assert "Fix:" in message


def test_subprocess_stdin_guard_rejects_undefaulted_kwargs_forwarding() -> None:
    """A wrapper that only forwards kwargs proves nothing about its callers."""
    source = (
        "import asyncio\n"
        "async def wrapper(*args, **kwargs):\n"
        "    return await asyncio.create_subprocess_exec(*args, **kwargs)\n"
    )

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    message = str(exc_info.value)
    assert "src/chrys/foundation/bad_probe.py:3" in message
    assert "AGENTS.md:39" in message


@pytest.mark.parametrize(
    ("label", "body"),
    [
        (
            "binding after the spawn",
            (
                "    proc = await asyncio.create_subprocess_exec(*args, **kwargs)\n"
                '    kwargs.setdefault("stdin", asyncio.subprocess.DEVNULL)\n'
                "    return proc\n"
            ),
        ),
        (
            "binding inside a branch",
            (
                "    if args:\n"
                '        kwargs.setdefault("stdin", asyncio.subprocess.DEVNULL)\n'
                "    return await asyncio.create_subprocess_exec(*args, **kwargs)\n"
            ),
        ),
        (
            "binding inside an unexecuted lambda",
            (
                '    later = lambda: kwargs.setdefault("stdin", asyncio.subprocess.DEVNULL)\n'
                "    return await asyncio.create_subprocess_exec(*args, **kwargs)\n"
            ),
        ),
    ],
)
def test_subprocess_stdin_guard_requires_the_binding_to_dominate(label: str, body: str) -> None:
    """Evidence gathered anywhere in the body would accept code that never runs."""
    source = "import asyncio\nasync def wrapper(*args, **kwargs):\n" + body

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    assert "AGENTS.md:39" in str(exc_info.value), label


@pytest.mark.parametrize(
    "binding",
    [
        'kwargs.setdefault("stdin", asyncio.subprocess.DEVNULL)',
        'kwargs["stdin"] = asyncio.subprocess.DEVNULL',
    ],
)
def test_subprocess_stdin_guard_accepts_forwarding_that_defaults_stdin(binding: str) -> None:
    """The splat is proof once the wrapper is seen binding stdin itself."""
    source = (
        "import asyncio\n"
        "async def wrapper(*args, **kwargs):\n"
        f"    {binding}\n"
        "    return await asyncio.create_subprocess_exec(*args, **kwargs)\n"
    )

    _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})


@pytest.mark.parametrize(
    "consumer",
    [
        "from chrys.app.tui.widgets import LocalizedWidget\nwidget = LocalizedWidget()\n",
        "from chrys.app.tui import widgets\nwidget = widgets.LocalizedWidget()\n",
        ("from chrys.app.tui.widgets.localized import LocalizedWidget as Widget\nwidget = Widget()\n"),
    ],
    ids=["package-import-bare-name", "module-qualified", "constructor-alias"],
)
def test_locale_controller_guard_covers_package_imports_qualification_and_aliases(consumer: str) -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/localized.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
        ),
        Path("src/chrys/app/tui/widgets/__init__.py"): (
            "from chrys.app.tui.widgets.localized import LocalizedWidget\n"
        ),
        Path("src/chrys/app/tui/screen.py"): consumer,
    }

    with pytest.raises(AssertionError, match=r"must forward locale_controller="):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_rejects_literal_none() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/localized.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
            "\n"
            "widget = LocalizedWidget(locale_controller=None)\n"
        )
    }

    with pytest.raises(AssertionError, match="must not pass literal locale_controller=None"):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_rejects_literal_none_on_super_call() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/localized.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
            "\n"
            "class SpecializedWidget(LocalizedWidget):\n"
            "    def __init__(self):\n"
            "        super().__init__(locale_controller=None)\n"
        )
    }

    with pytest.raises(AssertionError, match=r"__init__\(\.\.\.\) must not pass literal locale_controller=None"):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_rejects_positional_propagation() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/localized.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
            "\n"
            "widget = LocalizedWidget(controller)\n"
        )
    }

    with pytest.raises(AssertionError, match="must forward locale_controller= explicitly"):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


@pytest.mark.parametrize(
    "specialization",
    [
        (
            "from chrys.app.tui.widgets.base import LocalizedWidget\n"
            "class SpecializedWidget(LocalizedWidget):\n"
            "    pass\n"
        ),
        (
            "from chrys.app.tui.widgets.base import LocalizedWidget\n"
            "class IntermediateWidget(LocalizedWidget):\n"
            "    pass\n"
            "class SpecializedWidget(IntermediateWidget):\n"
            "    pass\n"
        ),
        (
            "from chrys.app.tui.widgets.base import LocalizedWidget as BaseWidget\n"
            "class SpecializedWidget(BaseWidget):\n"
            "    pass\n"
        ),
    ],
    ids=["direct", "transitive", "base-alias"],
)
def test_locale_controller_guard_covers_inherited_constructors(specialization: str) -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/base.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
        ),
        Path("src/chrys/app/tui/widgets/specialized.py"): specialization + "widget = SpecializedWidget()\n",
    }

    with pytest.raises(AssertionError, match=r"SpecializedWidget\(\.\.\.\) must forward locale_controller="):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_accepts_explicit_keyword_for_inherited_constructor() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/base.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
        ),
        Path("src/chrys/app/tui/widgets/specialized.py"): (
            "from chrys.app.tui.widgets.base import LocalizedWidget\n"
            "class SpecializedWidget(LocalizedWidget):\n"
            "    pass\n"
            "widget = SpecializedWidget(locale_controller=controller)\n"
        ),
    }

    sites = _locale_aware_tui_class_sites(sources)
    assert "SpecializedWidget" in sites
    _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_fails_closed_on_ambiguous_multiple_inheritance() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/base.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
            "\n"
            "class Mixin:\n"
            "    pass\n"
            "\n"
            "class SpecializedWidget(Mixin, LocalizedWidget):\n"
            "    pass\n"
        )
    }

    with pytest.raises(AssertionError, match="inherits locale context through multiple bases"):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_fails_closed_on_duplicate_class_names() -> None:
    locale_aware_constructor = (
        "class LocalizedWidget:\n"
        "    def __init__(self, *, locale_controller=None):\n"
        "        self.locale_controller = locale_controller\n"
    )
    sources = {
        Path("src/chrys/app/tui/one.py"): locale_aware_constructor,
        Path("src/chrys/app/tui/two.py"): "class LocalizedWidget:\n    pass\n",
    }

    with pytest.raises(AssertionError, match="locale-aware TUI class names must be unique"):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_fails_closed_on_duplicate_inherited_class_names() -> None:
    sources = {
        Path("src/chrys/app/tui/base.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
        ),
        Path("src/chrys/app/tui/one.py"): (
            "from chrys.app.tui.base import LocalizedWidget\nclass SpecializedWidget(LocalizedWidget):\n    pass\n"
        ),
        Path("src/chrys/app/tui/two.py"): "class SpecializedWidget:\n    pass\n",
    }

    with pytest.raises(AssertionError, match="duplicate definitions and inherits locale context"):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_aggregates_discovery_and_call_violations() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/localized.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
            "\n"
            "class Mixin:\n"
            "    pass\n"
            "\n"
            "class SpecializedWidget(Mixin, LocalizedWidget):\n"
            "    pass\n"
            "\n"
            "widget = LocalizedWidget()\n"
        )
    }

    with pytest.raises(AssertionError) as exc_info:
        _assert_tui_locale_controller_propagation_is_explicit(sources)

    message = str(exc_info.value)
    assert "inherits locale context through multiple bases" in message
    assert "LocalizedWidget(...) must forward locale_controller=" in message


def test_locale_controller_guard_accepts_dynamic_keyword_and_ignores_unrelated_constructors() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/localized.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
            "\n"
            "class PlainWidget:\n"
            "    pass\n"
        ),
        Path("src/chrys/app/tui/screen.py"): (
            "from chrys.app.tui.widgets.localized import LocalizedWidget as Widget, PlainWidget\n"
            "widget = Widget(locale_controller=getattr(app, 'locale_controller', None))\n"
            "plain = PlainWidget()\n"
        ),
    }

    _assert_tui_locale_controller_propagation_is_explicit(sources)


@_pins_allowlist("_EXCHANGE_WALKER_ALLOWLIST")
def test_exchange_walker_allowlist_entries_are_live() -> None:
    """Every justified walker entry must resolve to exactly ONE scope by its
    qualified name — a stale entry would silently vouch for a walker added
    later under the same name, and an ambiguous one for a scope it never
    justified."""
    sources = _src_sources()
    problems: list[str] = []
    for path, qualified in sorted(_EXCHANGE_WALKER_ALLOWLIST):
        source = sources.get(path)
        if source is None:
            problems.append(f"{path}: file missing")
            continue
        index = _ModuleScopes(_tree(path, source))
        matches = [scope for scope in index.scopes if index.qualified[id(scope)] == qualified]
        if not matches:
            problems.append(f"{path}: no scope named {qualified}")
        elif len(matches) > 1:
            problems.append(f"{path}: {qualified} is ambiguous ({len(matches)} scopes)")
    _tree.cache_clear()
    assert problems == [], "\n".join(problems)


@_pins_allowlist("_LOCAL_POLLING_ALLOWLIST")
def test_local_polling_allowlist_entries_are_live() -> None:
    trees = _allowlist_target_trees()
    problems: list[str] = []
    for path, helper_name in sorted(_LOCAL_POLLING_ALLOWLIST):
        tree = trees.get(path)
        if tree is None:
            problems.append(
                _meta_guard_problem(
                    "_LOCAL_POLLING_ALLOWLIST",
                    f"entry ({path!s}, {helper_name}) names a missing file",
                    "remove the stale entry or point it at the file that defines the approved polling helper",
                )
            )
            continue
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == helper_name
        ]
        if helper_name not in _POLLING_HELPER_NAMES:
            problems.append(
                _meta_guard_problem(
                    "_LOCAL_POLLING_ALLOWLIST",
                    f"entry ({path!s}, {helper_name}) cannot be consumed because the name is not a guarded polling helper",
                    "remove the entry or add the actual guarded helper name from _POLLING_HELPER_NAMES",
                )
            )
        elif not matches:
            problems.append(
                _meta_guard_problem(
                    "_LOCAL_POLLING_ALLOWLIST",
                    f"entry ({path!s}, {helper_name}) names no function in {path}",
                    "remove the stale entry or update it to the helper's current function name",
                )
            )
        elif len(matches) > 1:
            lines = ", ".join(str(node.lineno) for node in matches)
            problems.append(
                _meta_guard_problem(
                    "_LOCAL_POLLING_ALLOWLIST",
                    f"entry ({path!s}, {helper_name}) is ambiguous at {path}:{lines}",
                    "use a unique helper name or narrow the allowlist key before retaining the exemption",
                )
            )

    assert problems == [], "\n".join(problems)


@_pins_allowlist("_ENGINE_START_PATH_ALLOWLIST")
def test_engine_start_path_allowlist_entries_are_live() -> None:
    trees = _allowlist_target_trees()
    problems: list[str] = []
    for path in sorted(_ENGINE_START_PATH_ALLOWLIST):
        tree = trees.get(path)
        if tree is None:
            problems.append(
                _meta_guard_problem(
                    "_ENGINE_START_PATH_ALLOWLIST",
                    f"entry {path} names a missing file",
                    "remove the stale entry or update it to the live AgentEngine helper module",
                )
            )
            continue
        if not _direct_engine_start_lines(tree):
            problems.append(
                _meta_guard_problem(
                    "_ENGINE_START_PATH_ALLOWLIST",
                    f"entry {path} starts no locally constructed AgentEngine, so the guard would not flag it "
                    "even without the exemption",
                    "remove the stale entry or move it to the helper module that starts the engine directly",
                )
            )

    assert problems == [], "\n".join(problems)


@_pins_allowlist("_RAW_TOOL_LOOP_LAYER_ALLOWLIST")
def test_raw_tool_loop_layer_allowlist_entries_are_live() -> None:
    trees = _allowlist_target_trees()
    problems: list[str] = []
    for path, scope in sorted(_RAW_TOOL_LOOP_LAYER_ALLOWLIST):
        tree = trees.get(path)
        if tree is None:
            problems.append(
                _meta_guard_problem(
                    "_RAW_TOOL_LOOP_LAYER_ALLOWLIST",
                    f"entry ({path!s}, {scope}) names a missing file",
                    "remove the stale entry or point it at the live raw ToolLoopLayer scope",
                )
            )
            continue
        definitions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == scope
        ]
        matches = [(line, kind) for line, found_scope, kind in _raw_tool_loop_layer_uses(tree) if found_scope == scope]
        if not definitions:
            problems.append(
                _meta_guard_problem(
                    "_RAW_TOOL_LOOP_LAYER_ALLOWLIST",
                    f"entry ({path!s}, {scope}) names no test function or class in {path}",
                    "remove the stale entry or update it to the exact live test function or class name",
                )
            )
        elif len(definitions) > 1:
            sites = ", ".join(f"{path}:{node.lineno}" for node in definitions)
            problems.append(
                _meta_guard_problem(
                    "_RAW_TOOL_LOOP_LAYER_ALLOWLIST",
                    f"entry ({path!s}, {scope}) ambiguously resolves to multiple scopes: {sites}",
                    "rename the scopes or narrow the allowlist key so the entry resolves to exactly one symbol",
                )
            )
        elif not matches:
            problems.append(
                _meta_guard_problem(
                    "_RAW_TOOL_LOOP_LAYER_ALLOWLIST",
                    f"entry ({path!s}, {scope}) resolves to a scope with no raw ToolLoopLayer construction or subclass",
                    "remove the stale entry or update it to the live scope that directly uses ToolLoopLayer",
                )
            )

    assert problems == [], "\n".join(problems)


@_pins_allowlist("_DIRECT_TRAJECTORY_PREFIX_ORACLE_ALLOWLIST")
def test_direct_trajectory_prefix_oracle_allowlist_entries_are_live() -> None:
    trees = _allowlist_target_trees()
    problems: list[str] = []
    for path in sorted(_DIRECT_TRAJECTORY_PREFIX_ORACLE_ALLOWLIST):
        tree = trees.get(path)
        if tree is None:
            problems.append(
                _meta_guard_problem(
                    "_DIRECT_TRAJECTORY_PREFIX_ORACLE_ALLOWLIST",
                    f"entry {path} names a missing file",
                    "remove the stale entry or update it to the live direct prefix-oracle module",
                )
            )
            continue
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _qualified_name(node.func).endswith("verify_accounted_prefix")
        ]
        if not calls:
            problems.append(
                _meta_guard_problem(
                    "_DIRECT_TRAJECTORY_PREFIX_ORACLE_ALLOWLIST",
                    f"entry {path} contains no verify_accounted_prefix call and no longer consumes the exemption",
                    "remove the stale entry or update it to the module that directly calls verify_accounted_prefix",
                )
            )

    assert problems == [], "\n".join(problems)


@_pins_allowlist("_CLASSIFIER_IMPORT_ALLOWLIST")
def test_classifier_import_allowlist_entries_are_live() -> None:
    trees = _allowlist_target_trees()
    problems: list[str] = []
    for path in sorted(_CLASSIFIER_IMPORT_ALLOWLIST):
        tree = trees.get(path)
        if tree is None:
            problems.append(
                _meta_guard_problem(
                    "_CLASSIFIER_IMPORT_ALLOWLIST",
                    f"entry {path} names a missing file",
                    "remove the stale entry or update it to the live result-only-classifier importer",
                )
            )
            continue
        references = [
            node
            for node in ast.walk(tree)
            if (isinstance(node, ast.ImportFrom) and any(alias.name == _RESULT_ONLY_CLASSIFIER for alias in node.names))
            or (isinstance(node, ast.Attribute) and node.attr == _RESULT_ONLY_CLASSIFIER)
        ]
        if not references:
            problems.append(
                _meta_guard_problem(
                    "_CLASSIFIER_IMPORT_ALLOWLIST",
                    f"entry {path} contains no {_RESULT_ONLY_CLASSIFIER} import or qualified use",
                    "remove the stale entry or update it to the module that reaches the result-only classifier",
                )
            )

    assert problems == [], "\n".join(problems)


def test_scroll_relative_guard_rejects_counterexample() -> None:
    with pytest.raises(AssertionError, match="scroll_relative"):
        _assert_no_scroll_relative({Path("tests/test_bad.py"): "panel.scroll_relative(y=1)\n"})


def test_local_polling_guard_rejects_counterexample() -> None:
    with pytest.raises(AssertionError, match=r"tests/support/waiting\.py"):
        _assert_no_unapproved_local_polling_helpers(
            {Path("tests/test_bad.py"): "async def _eventually(predicate):\n    return predicate()\n"}
        )


@pytest.mark.parametrize(
    "source",
    [
        "from tests.support.waiting import wait_until\nasync def test_bad():\n    await wait_until(lambda: True)\n",
        "import tests.support.waiting as waits\nasync def test_bad():\n    await waits.wait_until(lambda: True)\n",
        (
            "import tests.support.waiting\n"
            "async def test_bad():\n"
            "    await tests.support.waiting.wait_until(lambda: True)\n"
        ),
        (
            "import tests.support.waiting\n"
            "import tests.support.engines\n"
            "async def test_bad():\n"
            "    await tests.support.waiting.wait_until(lambda: True)\n"
        ),
        "from tests.support import waiting\nasync def test_bad():\n    await waiting.wait_until(lambda: True)\n",
        (
            "async def test_bad():\n"
            "    from tests.support.waiting import wait_until\n"
            "    await wait_until(lambda: True)\n"
        ),
        (
            "from tests.support.waiting import wait_until\n"
            "async def outer():\n"
            "    async def test_bad():\n"
            "        await wait_until(lambda: True)\n"
        ),
    ],
)
def test_ignored_wait_until_guard_rejects_counterexamples(source: str) -> None:
    with pytest.raises(AssertionError, match="ignored wait_until result"):
        _assert_no_ignored_wait_until_results({Path("tests/test_bad.py"): source})


@pytest.mark.parametrize(
    ("path", "relative_import"),
    [
        (Path("tests/kernel/test_bad.py"), "from ..support.waiting import wait_until"),
        (Path("tests/service/context/test_bad.py"), "from ...support.waiting import wait_until"),
    ],
)
def test_ignored_wait_until_guard_resolves_relative_imports(path: Path, relative_import: str) -> None:
    source = f"{relative_import}\nasync def test_bad():\n    await wait_until(lambda: True)\n"

    with pytest.raises(AssertionError, match="ignored wait_until result"):
        _assert_no_ignored_wait_until_results({path: source})


def test_ignored_wait_until_guard_allows_consumed_results() -> None:
    source = (
        "from tests.support.waiting import wait_until\n"
        "async def test_good():\n"
        "    assert await wait_until(lambda: True)\n"
        "    observed = await wait_until(lambda: True)\n"
        "    return observed\n"
    )
    _assert_no_ignored_wait_until_results({Path("tests/test_good.py"): source})


def test_ignored_wait_until_guard_allows_consumed_relative_import_result() -> None:
    source = (
        "from ..support.waiting import wait_until\nasync def test_good():\n    assert await wait_until(lambda: True)\n"
    )

    _assert_no_ignored_wait_until_results({Path("tests/kernel/test_good.py"): source})


@pytest.mark.parametrize(
    "source",
    [
        ("from tests.support.waiting import wait_until\nasync def helper(wait_until):\n    await wait_until()\n"),
        (
            "from tests.support.waiting import wait_until\n"
            "async def helper():\n"
            "    wait_until = callback\n"
            "    await wait_until()\n"
        ),
        (
            "from tests.support.waiting import wait_until\n"
            "async def helper():\n"
            "    async def wait_until():\n"
            "        pass\n"
            "    await wait_until()\n"
        ),
        (
            "from tests.support.waiting import wait_until\n"
            "async def helper():\n"
            "    await wait_until()\n"
            "    from other import wait_until\n"
        ),
        (
            "from tests.support.waiting import wait_until\n"
            "from other import wait_until\n"
            "async def helper():\n"
            "    await wait_until()\n"
        ),
        ("from tests.support import waiting\nasync def helper(waiting):\n    await waiting.wait_until()\n"),
        (
            "import tests.support.waiting\n"
            "import other as tests\n"
            "async def helper():\n"
            "    await tests.support.waiting.wait_until()\n"
        ),
    ],
)
def test_ignored_wait_until_guard_honors_lexical_shadowing(source: str) -> None:
    _assert_no_ignored_wait_until_results({Path("tests/test_good.py"): source})


def test_tool_loop_layer_guard_rejects_counterexample() -> None:
    source = "def test_bad():\n    layer = ToolLoopLayer(ChatMiddlewareLayer(wire))\n"
    with pytest.raises(AssertionError, match="InvariantCheckedToolLoopLayer"):
        _assert_tool_loop_layer_uses_are_invariant_checked({Path("tests/test_bad.py"): source})


def test_tool_loop_layer_guard_attributes_calls_to_nearest_enclosing_function() -> None:
    # A construction inside a nested helper must be attributed to the helper,
    # not the (possibly allowlisted) outer test — otherwise an allowlist entry
    # would silently cover raw constructions it never vouched for.
    source = "def test_ctor_defaults_mirror_framework_values():\n    def _build():\n        return ToolLoopLayer(inner)\n    return _build()\n"
    with pytest.raises(AssertionError, match="InvariantCheckedToolLoopLayer"):
        _assert_tool_loop_layer_uses_are_invariant_checked({Path("tests/kernel/test_loop.py"): source})


def test_tool_loop_layer_guard_rejects_import_alias_counterexample() -> None:
    # Renaming the import must not launder a raw construction.
    source = "from chrys.kernel.loop import ToolLoopLayer as RawLoop\n\ndef test_bad():\n    layer = RawLoop(inner)\n"
    with pytest.raises(AssertionError, match="InvariantCheckedToolLoopLayer"):
        _assert_tool_loop_layer_uses_are_invariant_checked({Path("tests/test_bad.py"): source})


def test_tool_loop_layer_guard_rejects_subclass_counterexample() -> None:
    # Deriving from the raw layer runs the real loop with no oracle on its
    # final responses — the subclass form must be flagged like a construction.
    source = "class _Client(ToolLoopLayer):\n    pass\n"
    with pytest.raises(AssertionError, match="subclass InvariantCheckedToolLoopLayer"):
        _assert_tool_loop_layer_uses_are_invariant_checked({Path("tests/test_bad.py"): source})


def test_trajectory_prefix_guard_rejects_a_decoded_events_check() -> None:
    source = "def test_bad(result):\n    assert verify_accounted_prefix(result.events) == []\n"
    with pytest.raises(AssertionError, match="physical slots"):
        _assert_trajectory_prefix_checks_use_physical_slots({Path("tests/test_bad.py"): source})


def test_integration_marker_directory_guard_rejects_counterexample() -> None:
    source = "import pytest\n\n@pytest.mark.integration\ndef test_offline_cross_layer():\n    pass\n"
    with pytest.raises(AssertionError, match="offline cross-layer"):
        _assert_integration_marker_directory_disjoint({Path("tests/integration/test_bad.py"): source})


@pytest.mark.parametrize(
    ("imports", "constructor"),
    [
        ("", "AgentEngine"),
        ("import chrys.orchestration.engine.engine as engine_module", "engine_module.AgentEngine"),
        ("from chrys.orchestration.engine.engine import AgentEngine as AE", "AE"),
    ],
)
def test_direct_agent_engine_start_guard_rejects_counterexample(imports: str, constructor: str) -> None:
    source = f"{imports}\n\nasync def test_bad():\n    engine = {constructor}(bus)\n    await engine.start(profile)\n"
    with pytest.raises(AssertionError, match="agent_engine fixture"):
        _assert_no_direct_agent_engine_start({Path("tests/test_bad.py"): source})


@pytest.mark.parametrize(
    "marker",
    [
        '@pytest.mark.quarantine(expires="2099-01-01")',
        '@pytest.mark.quarantine(reason="known flake", expires="2099-1-1")',
        "@pytest.mark.quarantine",
    ],
)
def test_quarantine_metadata_guard_rejects_counterexamples(marker: str) -> None:
    source = f"import pytest\n\n{marker}\ndef test_bad():\n    pass\n"
    with pytest.raises(AssertionError, match="quarantine"):
        _assert_quarantine_marker_metadata({Path("tests/test_bad.py"): source})


def test_quarantine_expiry_guard_rejects_expired_counterexample() -> None:
    marker = '@pytest.mark.quarantine(reason="known flake", expires="2026-07-15")'
    source = f"import pytest\n\n{marker}\ndef test_bad():\n    pass\n"
    with pytest.raises(AssertionError, match="quarantine expired on 2026-07-15"):
        _assert_quarantine_marker_metadata({Path("tests/test_bad.py"): source}, today=date(2026, 7, 16))


def test_exchange_walker_guard_rejects_marker_forgetting_walker() -> None:
    # The most dangerous new walker FORGETS markers entirely, so no
    # marker-reference trigger can catch it — the walking constructs
    # themselves (role loop + type classification + pending state) must.
    source = (
        "def collect_unanswered(messages):\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if message.get("role") == "assistant":\n'
        '            for content in message.get("contents", []):\n'
        '                if content.get("type") == "function_call":\n'
        '                    pending[content.get("call_id")] = content\n'
        '        elif message.get("role") == "tool":\n'
        '            for content in message.get("contents", []):\n'
        '                if content.get("type") == "function_result":\n'
        '                    pending.pop(content.get("call_id"), None)\n'
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_helper_indirected_walker() -> None:
    # A cursor walk that classifies through a local helper never reads the
    # type sets in its own loop; the helper must not launder it.
    source = (
        "def _is_result_block(message):\n"
        '    return message.role == "tool" or any(\n'
        '        content.type == "function_result" for content in message.contents\n'
        "    )\n"
        "\n"
        "def find_block_end(messages, start):\n"
        "    index = start\n"
        "    while index < len(messages) and messages[index].role != 'user':\n"
        "        if not _is_result_block(messages[index]):\n"
        "            break\n"
        "        index += 1\n"
        "    return index\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_aliased_cursor_walker() -> None:
    # ``message = messages[index]`` inside a cursor walk is the ordinary
    # walker spelling; the alias must not hide the role read.
    source = (
        "def find_output_end(messages, start):\n"
        "    index = start\n"
        "    while index < len(messages):\n"
        "        message = messages[index]\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        index += 1\n"
        "    return index\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_loop_target_alias() -> None:
    # Rebinding the loop target to a fresh name walks the same items.
    source = (
        "def collect_calls(messages):\n"
        "    pending = {}\n"
        "    for item in messages:\n"
        "        message = item\n"
        '        if message.role == "assistant":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_call":\n'
        "                    pending[content.call_id] = content\n"
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_serialized_subscript_walker() -> None:
    # A dict-transcript walker reads role and call_id through subscripts.
    source = (
        "def collect_unanswered(messages):\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if message["role"] == "assistant":\n'
        '            for content in message["contents"]:\n'
        '                if content["type"] == "function_call":\n'
        '                    pending[content["call_id"]] = content\n'
        '        elif message["role"] == "tool":\n'
        '            for content in message["contents"]:\n'
        '                if content["type"] == "function_result":\n'
        '                    pending.pop(content["call_id"], None)\n'
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_dead_grammar_name_reference() -> None:
    # Merely naming iter_exchanges is not consuming it; only a call is.
    source = (
        "def collect_unanswered(messages):\n"
        "    unused = iter_exchanges\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if message.role == "assistant":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_call":\n'
        "                    pending[content.call_id] = content\n"
        '        elif message.role == "tool":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_result":\n'
        "                    pending.pop(content.call_id, None)\n"
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_role_helper_walker() -> None:
    # Factoring the role read into a local helper is ordinary refactoring,
    # not sanitization; calling it on a walked item is still a role read.
    source = (
        "def _role(message):\n"
        "    return message.role\n"
        "\n"
        "def collect_calls(messages):\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if _role(message) == "assistant":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_call":\n'
        "                    pending[content.call_id] = content\n"
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_id_helper_walker() -> None:
    # The pairing-state read hides inside a local key helper; the walk in
    # the caller is still keyed on call identity.
    source = (
        "def _call_key(content):\n"
        "    return content.call_id\n"
        "\n"
        "def collect_unanswered(messages):\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        "        for content in message.contents:\n"
        '            if content.type == "function_call":\n'
        "                pending[_call_key(content)] = content\n"
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_range_cursor_walker() -> None:
    # ``for index in range(...)`` over a subscripted sequence is the same
    # cursor walk as a while loop and must count as boundary state.
    source = (
        "def find_block_end(messages, start):\n"
        "    end = start\n"
        "    for index in range(start, len(messages)):\n"
        "        message = messages[index]\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_enumerate_cursor_walker() -> None:
    # ``for index, message in enumerate(...)`` doing index arithmetic is the
    # same cursor walk as a subscripted range loop — the exclusive-end
    # bookkeeping is boundary state even without a subscript.
    source = (
        "def find_block_end(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index + 1\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_enumerate_index_alias_arithmetic() -> None:
    # Copying the enumerate index into a fresh name before the arithmetic
    # is the same cursor walk.
    source = (
        "def find_block_end(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return end + 1\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_one_based_enumerate_cursor_walker() -> None:
    # ``enumerate(messages, 1)`` folds the ``+1`` into the start argument;
    # returning the tracked index is the same exclusive-end bookkeeping.
    source = (
        "def find_block_end(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages, 1):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_returned_enumerate_index_walker() -> None:
    # Returning the zero-based tracked index is the inclusive-end twin of
    # the one-based walk — the arithmetic just moves to the caller.
    source = (
        "def find_last_result_index(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_max_tracked_enumerate_index() -> None:
    # ``end = max(end, index)`` derives boundary state from the index
    # without any BinOp or direct copy — still a cursor walk.
    source = (
        "def find_last_result_index(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = max(end, index)\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_conditional_enumerate_index_capture() -> None:
    # A conditional-expression update is the same scalar boundary capture.
    source = (
        "def find_last_result_index(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        end = index if message.role == "tool" else end\n'
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_attribute_stored_enumerate_index() -> None:
    # Storing the index on a result object is structured boundary capture.
    source = (
        "def find_block_bounds(messages, bounds):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        bounds.end = index\n"
        "    return bounds\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_one_based_enumerate_reporter() -> None:
    # An explicit enumerate start adds no pairing state by itself: a
    # human-facing one-based reporter must stay out of reach.
    source = (
        "def call_message_positions(messages):\n"
        "    positions = []\n"
        "    for position, message in enumerate(messages, 1):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        "            positions.append(position)\n"
        "    return positions\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_accepts_enumerate_membership_test() -> None:
    # A membership test consumes the index as a question, not a value —
    # the sidecar-suppression shape in the TUI merge walk. The boolean
    # answer carries no cursor state.
    source = (
        "def suppress_sidecars(messages, sidecars):\n"
        "    kept = []\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        "            text = None if index in sidecars else message.text\n"
        "            kept.append(text)\n"
        "    return kept\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_index_boundary_comparison_walker() -> None:
    # ``index == len(messages) - 1`` never stores a cursor — the boolean
    # itself controls the exchange boundary. Equality/order comparisons on
    # the index are boundary logic; only membership tests stay exempt.
    source = (
        "def last_message_if_result_tail(messages):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        if index == len(messages) - 1:\n"
        "            return message\n"
        "    return None\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_tuple_unpacked_enumerate_index() -> None:
    # Parallel bounds updates are ordinary Python; the tuple target must
    # not hide the scalar capture.
    source = (
        "def find_block_end(messages):\n"
        "    previous_end = end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        previous_end, end = end, index\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_fixed_key_bounds_store() -> None:
    # A fixed-key bounds dictionary is serialized cursor state, not a
    # dynamic coordinate/reporter map.
    source = (
        "def find_block_bounds(messages, bounds):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        '        bounds["end"] = index\n'
        "    return bounds\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_formatted_label_reporter() -> None:
    # Formatting the index into display text consumes it without capturing
    # cursor state — the refactored twin of the direct append reporter.
    source = (
        "def call_message_labels(messages):\n"
        "    labels = []\n"
        "    for position, message in enumerate(messages, 1):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        '            label = f"message {position}"\n'
        "            labels.append(label)\n"
        "    return labels\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_index_truthiness_boundary_walker() -> None:
    # ``if not index`` is the first-boundary check spelled as truthiness —
    # the test gates the walk without storing or comparing explicitly.
    source = (
        "def first_message_if_result_head(messages):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        if not index:\n"
        "            return message\n"
        "    return None\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_match_case_boundary_walker() -> None:
    # ``match index: case 0`` is the same first-boundary gate in pattern
    # form.
    source = (
        "def first_message_if_result_head(messages):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        match index:\n"
        "            case 0:\n"
        "                return message\n"
        "    return None\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


@pytest.mark.parametrize("key", ["-1", '("end", 0)'])
def test_exchange_walker_guard_rejects_literal_key_bounds_store(key: str) -> None:
    # Fixed keys come in more AST shapes than a bare constant: a negative
    # index and a tuple of literals serialize cursor state all the same.
    source = (
        "def find_block_bounds(messages, bounds):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        f"        bounds[{key}] = index\n"
        "    return bounds\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_keyword_argument_bounds_store() -> None:
    # ``bounds.update(end=index)`` is fixed-field boundary storage through
    # a mutation call, not a dynamic reporter map.
    source = (
        "def find_block_bounds(messages, bounds):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        bounds.update(end=index)\n"
        "    return bounds\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_inline_formatted_comparison() -> None:
    # An f-string compared inline is the one-statement twin of the
    # accepted formatted-label reporter — rendering stays exempt inside a
    # comparison too.
    source = (
        "def find_selected_label(messages, selected_label):\n"
        "    labels = []\n"
        "    for position, message in enumerate(messages, 1):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        '            if f"message {position}" == selected_label:\n'
        "                labels.append(selected_label)\n"
        "    return labels\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


@pytest.mark.parametrize(
    "store",
    [
        'bounds.update({"end": index})',
        'bounds.setdefault("end", index)',
        'setattr(bounds, "end", index)',
        "bounds.update(dict(end=index))",
        'bounds.update([("end", index)])',
        'bounds.update({"start": 0}, end=index)',
        'bounds.update({"start": 0} | defaults, end=index)',
    ],
)
def test_exchange_walker_guard_rejects_positional_store_calls(store: str) -> None:
    # update/setdefault/setattr are the call spellings of a fixed-key
    # store — the same cursor state ``bounds["end"] = index`` serializes.
    source = (
        "def find_block_bounds(messages, bounds):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        f"        {store}\n"
        "    return bounds\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


@pytest.mark.parametrize(
    "report",
    ["reporter.emit(position=position)", 'logger.info("call site", extra={"position": position})'],
)
def test_exchange_walker_guard_accepts_keyword_reporting_calls(report: str) -> None:
    # Handing the position to another component by keyword is reporting,
    # not storage — only fixed-field store callees write cursor state.
    source = (
        "def report_call_positions(messages, reporter, logger):\n"
        "    for position, message in enumerate(messages, 1):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        f"            {report}\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_accepts_dynamic_key_store_call() -> None:
    # A dynamic key keeps the coordinate-map exemption through the call
    # spelling of a store too.
    source = (
        "def positions_by_message(messages, positions):\n"
        "    for position, message in enumerate(messages, 1):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        "            positions.setdefault(message.id, position)\n"
        "    return positions\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_match_guard_boundary_walker() -> None:
    # A case guard gates the boundary exactly like an if test — the same
    # cursor question moved from the subject into the guard.
    source = (
        "def first_message_if_result_head(messages):\n"
        "    for index, message in enumerate(messages):\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        match message.role:\n"
        '            case "tool" if not index:\n'
        "                return message\n"
        "    return None\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


@pytest.mark.parametrize("gate", ["not index", "index == len(messages) - 1"])
def test_exchange_walker_guard_rejects_comprehension_boundary_gate(gate: str) -> None:
    # Comprehension filters gate the walk exactly like statement-loop
    # tests; an enumerate cursor does not launder through comprehension
    # form.
    source = (
        "def result_heads(messages):\n"
        "    return [\n"
        "        message\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        f"        if {gate}\n"
        "    ]\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_comprehension_position_reporter() -> None:
    # A comprehension whose value slot collects positions is the
    # comprehension spelling of the append reporter.
    source = (
        "def call_positions(messages):\n"
        "    return [\n"
        "        position\n"
        "        for position, message in enumerate(messages, 1)\n"
        '        if message.role == "assistant"\n'
        '        if any(content.type == "function_call" for content in message.contents)\n'
        "    ]\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_collapsed_comprehension_cursor() -> None:
    # max() over a comprehension of indices collapses the container into
    # scalar cursor state — the reporter exemption stops at collapse.
    source = (
        "def last_result_index(messages):\n"
        "    end = max(\n"
        "        index\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        "    )\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_positional_update_reporter() -> None:
    # ``progress.update(task_id, completed=position)`` carries a positional
    # handle argument — the reporter-API shape, not the keyword-only or
    # mapping-shaped dict.update store idioms.
    source = (
        "def report_call_positions(messages, progress, task_id):\n"
        "    for position, message in enumerate(messages, 1):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        "            progress.update(task_id, completed=position)\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_subscript_collapsed_comprehension() -> None:
    # Subscripting a comprehension of indices extracts scalar cursor state
    # in one expression — the reporter exemption stops at collapse.
    source = (
        "def last_result_index(messages):\n"
        "    return [\n"
        "        index\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        "    ][-1]\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_sliced_comprehension_report() -> None:
    # A slice keeps the container a container — a truncated positions
    # report, not a scalar extraction.
    source = (
        "def first_result_positions(messages):\n"
        "    return [\n"
        "        index\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        "    ][:3]\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_accepts_outer_name_shadowed_by_comprehension_index() -> None:
    # Python scopes a comprehension target to the comprehension; an outer
    # parameter that happens to share its name is not an enumerate cursor.
    source = (
        "def report_call_positions(messages, index):\n"
        "    positions = [\n"
        "        index\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "assistant"\n'
        '        if any(content.type == "function_call" for content in message.contents)\n'
        "    ]\n"
        "    if index:\n"
        "        return positions\n"
        "    return []\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_lambda_comprehension_walker() -> None:
    # A lambda is a scope of its own and gets its own evaluation — a
    # factory-returned selector cannot launder a comprehension walk.
    source = (
        "def result_head_selector():\n"
        "    return lambda messages: [\n"
        "        message\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        "        if not index\n"
        "    ][0]\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_shadowed_grammar_helper_name() -> None:
    # A walker's own nested helper SHADOWS a module-level grammar helper
    # of the same name — the call resolves to the nearest binding, which
    # consumes no grammar.
    source = (
        "def block_pairs(messages):\n"
        "    return list(pair_results(messages))\n"
        "\n"
        "\n"
        "def find_last_result(messages):\n"
        "    def block_pairs():\n"
        "        return []\n"
        "    pairs = block_pairs()\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return pairs, end\n"
    )
    with pytest.raises(AssertionError, match="find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_cross_class_self_helper() -> None:
    # ``self.block_pairs`` resolves within the calling method's OWN class;
    # a grammar-backed method of another class does not exempt it.
    source = (
        "class GrammarConsumer:\n"
        "    def block_pairs(self, messages):\n"
        "        return list(pair_results(messages))\n"
        "\n"
        "\n"
        "class Walker:\n"
        "    def block_pairs(self, messages):\n"
        "        return []\n"
        "\n"
        "    def find_last_result(self, messages):\n"
        "        pairs = self.block_pairs(messages)\n"
        "        end = 0\n"
        "        for index, message in enumerate(messages):\n"
        '            if message.role != "tool":\n'
        "                break\n"
        '            if not any(content.type == "function_result" for content in message.contents):\n'
        "                break\n"
        "            end = index\n"
        "        return pairs, end\n"
    )
    with pytest.raises(AssertionError, match=r"Walker\.find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_scopes_allowlist_entries_to_their_class() -> None:
    # An allowlist entry vouches for ONE scope; a same-named method on a
    # different class in the same file is not covered.
    source = (
        "class NonVisionImageStubMiddleware:\n"
        "    def process(self, messages):\n"
        "        return messages\n"
        "\n"
        "\n"
        "class NewWalker:\n"
        "    def process(self, messages):\n"
        "        end = 0\n"
        "        for index, message in enumerate(messages):\n"
        '            if message.role != "tool":\n'
        "                break\n"
        '            if not any(content.type == "function_result" for content in message.contents):\n'
        "                break\n"
        "            end = index\n"
        "        return end\n"
    )
    with pytest.raises(AssertionError, match=r"NewWalker\.process hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/vision.py"): source})


def test_exchange_walker_guard_rejects_lambda_classifier_helper() -> None:
    # A name-bound lambda is a helper like any def — classification
    # through it propagates to the calling walker.
    source = (
        'is_result = lambda content: content.type == "function_result"\n'
        "\n"
        "\n"
        "def find_last_result(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        "        if not any(is_result(content) for content in message.contents):\n"
        "            break\n"
        "        end = index\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_lambda_role_helper() -> None:
    # Role reads through a name-bound lambda count the same as inline.
    source = (
        "read_role = lambda message: message.role\n"
        "\n"
        "\n"
        "def find_last_result(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if read_role(message) != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_singleton_unpacked_comprehension() -> None:
    # ``end, = [...]`` destructures the container away in one statement —
    # scalar extraction, not a positions report.
    source = (
        "def sole_result_index(messages):\n"
        "    end, = [\n"
        "        index\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        "    ]\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_parameter_shadowed_grammar_helper() -> None:
    # A parameter shadows a same-named module grammar helper — the walker
    # calls its CALLBACK, not the helper, and earns no exemption.
    source = (
        "def block_pairs(messages):\n"
        "    return list(pair_results(messages))\n"
        "\n"
        "\n"
        "def find_last_result(messages, block_pairs):\n"
        "    pairs = block_pairs(messages)\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return pairs, end\n"
    )
    with pytest.raises(AssertionError, match="find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_cross_class_attribute_grammar_exemption() -> None:
    # ``consumer.block_pairs`` cannot borrow another class's grammar-backed
    # method — with mixed same-named candidates the coarse attribute
    # fallback refuses to exempt.
    source = (
        "class GrammarConsumer:\n"
        "    def block_pairs(self, messages):\n"
        "        return list(pair_results(messages))\n"
        "\n"
        "\n"
        "class LocalConsumer:\n"
        "    def block_pairs(self, messages):\n"
        "        return []\n"
        "\n"
        "\n"
        "def find_last_result(messages, consumer: LocalConsumer):\n"
        "    pairs = consumer.block_pairs(messages)\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return pairs, end\n"
    )
    with pytest.raises(AssertionError, match="find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_sole_attribute_grammar_helper() -> None:
    # When every module binding of the called name is grammar-backed, an
    # attribute call through a collaborator keeps the exemption.
    source = (
        "class GrammarConsumer:\n"
        "    def block_pairs(self, messages):\n"
        "        return list(pair_results(messages))\n"
        "\n"
        "\n"
        "def find_last_result(messages, consumer):\n"
        "    pairs = consumer.block_pairs(messages)\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return pairs, end\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_inherited_role_helper() -> None:
    # ``self.read_role`` inherited from a same-module base class reaches
    # the walker — resolution climbs the local class lineage.
    source = (
        "class RoleReader:\n"
        "    def read_role(self, message):\n"
        "        return message.role\n"
        "\n"
        "\n"
        "class Walker(RoleReader):\n"
        "    def find_last_result(self, messages):\n"
        "        end = 0\n"
        "        for index, message in enumerate(messages):\n"
        '            if self.read_role(message) != "tool":\n'
        "                break\n"
        '            if not any(content.type == "function_result" for content in message.contents):\n'
        "                break\n"
        "            end = index\n"
        "        return end\n"
    )
    with pytest.raises(AssertionError, match=r"Walker\.find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_generic_base_inherited_role_helper() -> None:
    # A subscripted (generic) local base is still the same class — lineage
    # climbing unwraps the subscript.
    source = (
        "class RoleReader:\n"
        "    def read_role(self, message):\n"
        "        return message.role\n"
        "\n"
        "\n"
        "class Walker(RoleReader[Message]):\n"
        "    def find_last_result(self, messages):\n"
        "        end = 0\n"
        "        for index, message in enumerate(messages):\n"
        '            if self.read_role(message) != "tool":\n'
        "                break\n"
        '            if not any(content.type == "function_result" for content in message.contents):\n'
        "                break\n"
        "            end = index\n"
        "        return end\n"
    )
    with pytest.raises(AssertionError, match=r"Walker\.find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_reversed_enumerate_walker() -> None:
    # ``reversed(list(enumerate(...)))`` is the natural backwards scan —
    # wrapper calls don't hide the enumerate cursor.
    source = (
        "def find_last_result(messages):\n"
        "    for index, message in reversed(list(enumerate(messages))):\n"
        '        if message.role != "tool":\n'
        "            continue\n"
        '        if any(content.type == "function_result" for content in message.contents):\n'
        "            return index\n"
        "    return None\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_popped_comprehension_cursor() -> None:
    # ``[...].pop()`` collapses the container in the same statement —
    # the method-call spelling of the max/subscript collapse.
    source = (
        "def last_result_index(messages):\n"
        "    return [\n"
        "        index\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        "    ].pop()\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_cross_scope_helper_name_collision() -> None:
    # A grammar-backed helper nested in one function must not exempt a
    # walker in another function calling its OWN same-named helper —
    # bare-name propagation respects lexical visibility.
    source = (
        "def grammar_consumer(messages):\n"
        "    def block_pairs():\n"
        "        return list(pair_results(messages))\n"
        "    return block_pairs()\n"
        "\n"
        "\n"
        "def find_last_result(messages):\n"
        "    def block_pairs():\n"
        "        return []\n"
        "    pairs = block_pairs()\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return pairs, end\n"
    )
    with pytest.raises(AssertionError, match="find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_formatter_with_nested_indexed_helper() -> None:
    # A nested helper is its own scope: its enumerate cursor gets its own
    # guard evaluation and must not be attributed to the enclosing
    # formatter.
    source = (
        "def render_transcript(messages):\n"
        "    def numbered_lines(lines):\n"
        "        for index, line in enumerate(lines):\n"
        "            if index:\n"
        "                yield line\n"
        "    rendered = []\n"
        "    for message in messages:\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        "            rendered.extend(numbered_lines(message.text.splitlines()))\n"
        "    return rendered\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_keyword_role_helper_walker() -> None:
    # Passing the walked item by keyword is the same helper role read.
    source = (
        "def _role(message):\n"
        "    return message.role\n"
        "\n"
        "def collect_calls(messages):\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if _role(message=message) == "assistant":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_call":\n'
        "                    pending[content.call_id] = content\n"
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_discarded_grammar_call() -> None:
    # A grammar call whose result is thrown away consumes nothing; the
    # hand-rolled walk beside it is still the transcript authority.
    source = (
        "def collect_unanswered(messages, accessor):\n"
        "    iter_exchanges(messages, accessor)\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if message.role == "assistant":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_call":\n'
        "                    pending[content.call_id] = content\n"
        '        elif message.role == "tool":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_result":\n'
        "                    pending.pop(content.call_id, None)\n"
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_grammar_based_consumer() -> None:
    # A consumer on the shared grammar may still run a role-reading skeleton
    # walk (trim's backward walk) — the grammar reference is the tell.
    source = (
        "def trim(messages):\n"
        "    owners = {e.response_indices[0]: e for e in iter_exchanges(messages, ACCESSOR)}\n"
        "    i = len(messages) - 1\n"
        "    while i >= 0:\n"
        '        if messages[i].role != "assistant":\n'
        "            i -= 1\n"
        "            continue\n"
        '        if any(c.type == "function_call" and c.call_id for c in messages[i].contents):\n'
        "            break\n"
        "        i -= 1\n"
        "    return i\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_accepts_per_message_serializer() -> None:
    # A wire serializer reads role off its parameter and loops one message's
    # contents; it walks no transcript and must stay out of reach.
    source = (
        "def prepare_message(message):\n"
        "    items = []\n"
        "    for content in message.contents:\n"
        '        if message.role == "assistant" and content.type == "function_call":\n'
        '            items.append({"id": content.call_id})\n'
        "    return items\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_accepts_enumerate_without_index_arithmetic() -> None:
    # Enumerate alone is not pairing state: a reporter that collects indices
    # without arithmetic, subscripting, or id reads stays out of reach.
    source = (
        "def call_message_indices(messages):\n"
        "    indices = []\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        "            indices.append(index)\n"
        "    return indices\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_classifier_import_backstop_rejects_unallowlisted_module() -> None:
    source = "from chrys.kernel.exchanges import is_result_only_message\n"
    with pytest.raises(AssertionError, match="_CLASSIFIER_IMPORT_ALLOWLIST"):
        _assert_result_only_classifier_imports_are_allowlisted({Path("src/chrys/service/bad.py"): source})


def test_classifier_import_backstop_accepts_justified_importer() -> None:
    source = "from .exchanges import is_result_only_message\n"
    _assert_result_only_classifier_imports_are_allowlisted({Path("src/chrys/kernel/compaction.py"): source})


def test_classifier_backstop_rejects_module_alias_attribute_use() -> None:
    source = (
        "import chrys.kernel.exchanges as ex\n"
        "\n"
        "def check(message, accessor):\n"
        "    return ex.is_result_only_message(message, accessor)\n"
    )
    with pytest.raises(AssertionError, match="_CLASSIFIER_IMPORT_ALLOWLIST"):
        _assert_result_only_classifier_imports_are_allowlisted({Path("src/chrys/service/bad.py"): source})


@_pins_allowlist("_TUI_BINDING_CONSTRUCTION_ALLOWLIST")
def test_tui_binding_display_guard_allowlist_entries_are_live_and_unambiguous() -> None:
    observed: list[tuple[Path, str, str, str | None, str | None, str | None, bool | str]] = []
    for path, source in _src_sources().items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        tree = _tree(path, source)
        parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        binding_references = _binding_constructor_references(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                _qualified_name(node.func) in binding_references
                or _qualified_name(node.func).rsplit(".", maxsplit=1)[-1] == "Binding"
            ):
                owner = _binding_owner(node, parents)
                if path != _TUI_BINDING_DISPLAY_MODULE or owner != "localized_binding":
                    observed.append(_binding_site(path, owner, "Binding", node.args, node.keywords))
            value = _bindings_assignment_value(node)
            if not isinstance(value, (ast.List, ast.Tuple)):
                continue
            owner = _binding_owner(node, parents)
            observed.extend(
                _binding_site(path, owner, "tuple", item.elts)
                for item in value.elts
                if isinstance(item, ast.Tuple) and len(item.elts) in {2, 3}
            )

    assert len(_TUI_BINDING_CONSTRUCTION_ALLOWLIST) == 20
    assert len(observed) == len(set(observed))
    assert set(observed) == _TUI_BINDING_CONSTRUCTION_ALLOWLIST


@_pins_allowlist("_TUI_PROSE_SINK_ALLOWLIST")
def test_tui_prose_sink_allowlist_entries_are_live() -> None:
    sources = _src_sources()
    consumed: set[tuple[Path, str, str]] = set()
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        tree = _tree(path, source)
        for find_sites in _TUI_PROSE_SITE_FINDERS:
            for _node, sink_tag, _field, value in find_sites(tree):
                if not _is_tui_prose_bearing_literal(value, strip_substitutions=sink_tag == "from_markup"):
                    continue
                entry = _tui_prose_allowlist_entry(path, sink_tag, value)
                if entry is not None:
                    consumed.add(entry)

    for rule in (
        _assert_tui_notify_prose_is_localized,
        _assert_tui_border_titles_are_localized,
        _assert_tui_widget_label_prose_is_localized,
        _assert_tui_placeholder_tooltip_prose_is_localized,
    ):
        rule(sources)
    _tree.cache_clear()

    assert consumed == _TUI_PROSE_SINK_ALLOWLIST


def test_every_non_empty_allowlist_has_a_liveness_pin() -> None:
    # Empty allowlists have no entry that can go stale, so they need no pin.
    allowlists = {
        name: value for name, value in globals().items() if name.startswith("_") and name.endswith("_ALLOWLIST")
    }
    expected = {name for name, value in allowlists.items() if value}
    registered = set(_ALLOWLIST_PIN_REGISTRY)
    problems = [
        _meta_guard_problem(
            name,
            f"non-empty {name} has no registered *_allowlist_entries_are_live* test",
            f"decorate a focused liveness test with @_pins_allowlist({name!r}) and make it resolve every entry",
        )
        for name in sorted(expected - registered)
    ]
    for name in sorted(expected & registered):
        pins = _ALLOWLIST_PIN_REGISTRY[name]
        if not any(pin.__name__.startswith("test_") and "_allowlist_entries_are_live" in pin.__name__ for pin in pins):
            problems.append(
                _meta_guard_problem(
                    name,
                    f"{name} is registered only to tests that do not match *_allowlist_entries_are_live*",
                    "rename the registered test to describe its liveness contract or register the intended liveness test",
                )
            )
    for name in sorted(registered - set(allowlists)):
        pin_names = ", ".join(pin.__name__ for pin in _ALLOWLIST_PIN_REGISTRY[name])
        problems.append(
            _meta_guard_problem(
                name,
                f"{name} has liveness-pin registration ({pin_names}) but no matching global allowlist",
                "remove the stale registration or restore the allowlist global whose entries the test resolves",
            )
        )

    assert problems == [], "\n".join(problems)


@pytest.mark.parametrize(
    "source",
    [
        "from textual.binding import Binding\nBINDINGS = [Binding('x', 'do_thing', 'Do thing')]\n",
        "class Example:\n    BINDINGS = [('x', 'do_thing', 'Do thing')]\n",
        "app.bind('x', 'do_thing', description='Do thing')\n",
    ],
    ids=["direct-binding", "tuple-shorthand", "display-bind-call"],
)
def test_tui_binding_display_guard_rejects_noncanonical_shapes(source: str) -> None:
    with pytest.raises(AssertionError, match="localized_binding"):
        _assert_tui_binding_display_construction_is_canonical({Path("src/chrys/app/tui/screens/bad.py"): source})


def test_tui_binding_display_guard_accepts_localized_binding() -> None:
    source = (
        "from chrys.app.tui.binding_display import localized_binding\n"
        "BINDINGS = [localized_binding('x', 'do_thing', DEFINITION)]\n"
    )

    _assert_tui_binding_display_construction_is_canonical({Path("src/chrys/app/tui/screens/good.py"): source})


def test_tui_binding_display_guard_accepts_allowlisted_invisible_site_verbatim() -> None:
    source = (
        "from textual.binding import Binding\n"
        "class MainScreen:\n"
        "    BINDINGS = [Binding('ctrl+r', 'prompt_history', show=False, priority=True)]\n"
    )

    _assert_tui_binding_display_construction_is_canonical({Path("src/chrys/app/tui/screens/main/screen.py"): source})


@pytest.mark.parametrize(
    "binding_source",
    [
        "Binding('ctrl+r', 'prompt_history', show=True, priority=True)",
        "Binding('ctrl+r', 'prompt_history', 'NOW VISIBLE', show=False, priority=True)",
        "Binding('ctrl+r', 'prompt_history', show=SOME_FLAG, priority=True)",
    ],
    ids=["show-flip", "description-change", "dynamic-show"],
)
def test_tui_binding_display_guard_rejects_display_field_drift_at_allowlisted_site(
    binding_source: str,
) -> None:
    source = f"from textual.binding import Binding\nclass MainScreen:\n    BINDINGS = [{binding_source}]\n"

    with pytest.raises(AssertionError, match="localized_binding"):
        _assert_tui_binding_display_construction_is_canonical(
            {Path("src/chrys/app/tui/screens/main/screen.py"): source}
        )


@pytest.mark.parametrize(
    "source",
    [
        'self.notify("Connection failed")\n',
        'notify(message="Connection failed")\n',
        'notify(f"Loading {name}")\n',
        'self.notify(MESSAGE, title="Connection failed")\n',
        'self.notify("$Unlocalized")\n',
    ],
    ids=["positional-message", "keyword-message", "formatted-message", "title", "dollar-prose"],
)
def test_tui_notify_prose_guard_rejects_literals(source: str) -> None:
    with pytest.raises(AssertionError, match="localize raw prose"):
        _assert_tui_notify_prose_is_localized({Path("src/chrys/app/tui/screens/bad.py"): source})


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (Path("src/chrys/app/tui/screens/good.py"), 'notify(f"[red]*[/red] {value}")\n'),
        (Path("src/chrys/app/tui/screens/good.py"), 'notify("✕")\n'),
        (Path("src/chrys/app/tui/screens/good.py"), "notify(render_str(MESSAGE))\n"),
        (Path("src/chrys/app/cli/good.py"), 'notify("Raw English")\n'),
        (Path("src/chrys/app/tui/widgets/chat/messages.py"), 'Static("thinking")\n'),
        (Path("src/chrys/app/tui/screens/good.py"), 'notify("")\n'),
        (Path("src/chrys/app/tui/screens/good.py"), 'notify(MESSAGE, severity="warning")\n'),
    ],
    ids=["markup-fstring", "glyph", "render-call", "outside-tui", "allowlisted-site", "empty", "severity"],
)
def test_tui_notify_prose_guard_accepts_nonprose_and_nonsinks(path: Path, source: str) -> None:
    _assert_tui_notify_prose_is_localized({path: source})


@pytest.mark.parametrize(
    "source",
    [
        'Widget.border_title = "Connection details"\n',
        'widget.border_subtitle = f"Loading {name}"\n',
    ],
    ids=["title", "formatted-subtitle"],
)
def test_tui_border_title_prose_guard_rejects_literals(source: str) -> None:
    with pytest.raises(AssertionError, match="localize raw prose"):
        _assert_tui_border_titles_are_localized({Path("src/chrys/app/tui/screens/bad.py"): source})


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (Path("src/chrys/app/tui/screens/good.py"), 'widget.border_title = f"[red]*[/red] {value}"\n'),
        (Path("src/chrys/app/tui/screens/good.py"), 'widget.border_subtitle = "✕"\n'),
        (Path("src/chrys/app/tui/screens/good.py"), "widget.border_title = render_str(MESSAGE)\n"),
        (Path("src/chrys/app/cli/good.py"), 'widget.border_title = "Raw English"\n'),
        (Path("src/chrys/app/tui/widgets/chat/messages.py"), 'Static("thinking")\n'),
        (Path("src/chrys/app/tui/screens/good.py"), 'widget.border_title = ""\n'),
    ],
    ids=["markup-fstring", "glyph", "render-call", "outside-tui", "allowlisted-site", "empty"],
)
def test_tui_border_title_prose_guard_accepts_nonprose_and_nonsinks(path: Path, source: str) -> None:
    _assert_tui_border_titles_are_localized({path: source})


@pytest.mark.parametrize(
    "source",
    [
        'Label("Connection details")\n',
        'ui.Button(f"Loading {name}")\n',
        'Checkbox("Remember choice")\n',
        'TabPane("Session details")\n',
        'Static("Event stream")\n',
    ],
    ids=["label", "formatted-button", "checkbox", "tab-pane", "static"],
)
def test_tui_widget_label_prose_guard_rejects_literals(source: str) -> None:
    with pytest.raises(AssertionError, match="localize raw prose"):
        _assert_tui_widget_label_prose_is_localized({Path("src/chrys/app/tui/screens/bad.py"): source})


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (Path("src/chrys/app/tui/screens/good.py"), 'Label(f"[red]*[/red] {value}")\n'),
        (Path("src/chrys/app/tui/screens/good.py"), 'Button("✕")\n'),
        (Path("src/chrys/app/tui/screens/good.py"), "Checkbox(render_str(MESSAGE))\n"),
        (Path("src/chrys/app/cli/good.py"), 'Static("Raw English")\n'),
        (Path("src/chrys/app/tui/widgets/chat/messages.py"), 'Static("thinking")\n'),
        (Path("src/chrys/app/tui/screens/good.py"), 'TabPane("")\n'),
        (Path("src/chrys/app/tui/screens/good.py"), 'Label(text="Raw English")\n'),
    ],
    ids=[
        "markup-fstring",
        "glyph",
        "render-call",
        "outside-tui",
        "allowlisted-site",
        "empty",
        "keyword-label",
    ],
)
def test_tui_widget_label_prose_guard_accepts_nonprose_and_nonsinks(path: Path, source: str) -> None:
    _assert_tui_widget_label_prose_is_localized({path: source})


@pytest.mark.parametrize(
    "source",
    [
        'Input(placeholder="Enter a name")\n',
        'Button("Details", tooltip="Open details")\n',
        'widget.placeholder = "Enter a name"\n',
        'widget.tooltip = f"Loading {name}"\n',
    ],
    ids=["placeholder-keyword", "tooltip-keyword", "placeholder-assignment", "formatted-tooltip-assignment"],
)
def test_tui_placeholder_tooltip_prose_guard_rejects_literals(source: str) -> None:
    with pytest.raises(AssertionError, match="localize raw prose"):
        _assert_tui_placeholder_tooltip_prose_is_localized({Path("src/chrys/app/tui/screens/bad.py"): source})


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (Path("src/chrys/app/tui/screens/good.py"), 'Input(placeholder=f"[red]*[/red] {value}")\n'),
        (Path("src/chrys/app/tui/screens/good.py"), 'widget.tooltip = "✕"\n'),
        (Path("src/chrys/app/tui/screens/good.py"), "Input(tooltip=render_str(MESSAGE))\n"),
        (Path("src/chrys/app/cli/good.py"), 'Input(placeholder="Raw English")\n'),
        (Path("src/chrys/app/tui/screens/models/screen.py"), 'Input(placeholder="sk-...")\n'),
        (Path("src/chrys/app/tui/screens/good.py"), 'widget.placeholder = ""\n'),
        (Path("src/chrys/app/tui/screens/good.py"), 'placeholder = "Raw English"\n'),
    ],
    ids=[
        "markup-fstring",
        "glyph",
        "render-call",
        "outside-tui",
        "allowlisted-site",
        "empty",
        "plain-name-assignment",
    ],
)
def test_tui_placeholder_tooltip_prose_guard_accepts_nonprose_and_nonsinks(path: Path, source: str) -> None:
    _assert_tui_placeholder_tooltip_prose_is_localized({path: source})


@pytest.mark.parametrize(
    "source",
    [
        'Content.from_markup("[b]Compacting conversation...[/b]")\n',
        'Content.from_markup(f"[$error]✗ Compaction failed[/]{elapsed}")\n',
        'Text.from_markup("[red]Connection failed[/red]")\n',
        'Content.from_markup(markup="[b]Connection failed[/b]")\n',
        'Text.from_markup(text="Connection failed")\n',
    ],
    ids=["constant-prose", "fstring-prose", "text-prose", "markup-keyword", "text-keyword"],
)
def test_tui_content_markup_prose_guard_rejects_literals(source: str) -> None:
    with pytest.raises(AssertionError, match="localize raw prose"):
        _assert_tui_content_markup_prose_is_localized({Path("src/chrys/app/tui/widgets/bad.py"): source})


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (Path("src/chrys/app/tui/widgets/good.py"), 'Content.from_markup("[b]$label[/b]", label=label)\n'),
        (
            Path("src/chrys/app/tui/widgets/good.py"),
            'Content.from_markup(f"[b]{prefix}$label[/b] [$success]✓[/]{elapsed}", label=label)\n',
        ),
        (
            Path("src/chrys/app/tui/widgets/good.py"),
            'Content.from_markup("[$text-success][b]+$additions[/b][/]", additions=additions)\n',
        ),
        (Path("src/chrys/app/cli/good.py"), 'Content.from_markup("[b]Raw English[/b]")\n'),
        (Path("src/chrys/app/tui/widgets/good.py"), "Content.from_markup(template)\n"),
        (Path("src/chrys/app/tui/widgets/good.py"), 'Content.from_markup("[b]$text[/b]", text=value)\n'),
    ],
    ids=[
        "substitution-skeleton",
        "fstring-skeleton",
        "numeric-skeleton",
        "outside-tui",
        "nonliteral",
        "substitution-keyword-variable",
    ],
)
def test_tui_content_markup_prose_guard_accepts_skeletons_and_nonsinks(path: Path, source: str) -> None:
    _assert_tui_content_markup_prose_is_localized({path: source})


@pytest.mark.parametrize(
    "source",
    [
        "from textual.content import Content\nContent.from_text(value)\n",
        "from textual.content import Content as TContent\nTContent.from_text(value, markup=True)\n",
        "import textual.content as tc\ntc.Content.from_text(value, markup=allow_markup)\n",
    ],
    ids=["implicit-markup", "explicit-markup", "dynamic-markup"],
)
def test_tui_content_from_text_guard_rejects_markup_parsing(source: str) -> None:
    with pytest.raises(AssertionError, match="must pass markup=False"):
        _assert_tui_content_from_text_disables_markup({Path("src/chrys/app/tui/widgets/bad.py"): source})


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (
            Path("src/chrys/app/tui/widgets/good.py"),
            "from textual.content import Content\nContent.from_text(value, markup=False)\n",
        ),
        (
            Path("src/chrys/app/tui/widgets/good.py"),
            "from textual.content import Content\nContent.from_markup(template)\n",
        ),
        (
            Path("src/chrys/app/cli/good.py"),
            "from textual.content import Content\nContent.from_text(value)\n",
        ),
    ],
    ids=["literal-text", "intentional-markup", "outside-tui"],
)
def test_tui_content_from_text_guard_accepts_explicit_literal_text_and_nonsinks(path: Path, source: str) -> None:
    _assert_tui_content_from_text_disables_markup({path: source})


def test_tui_prose_sink_allowlist_rejects_literal_drift() -> None:
    source = 'Static("pondering")\n'
    with pytest.raises(AssertionError, match="Static"):
        _assert_tui_widget_label_prose_is_localized({Path("src/chrys/app/tui/widgets/chat/messages.py"): source})


def test_i18n_message_guard_accepts_canonical_definition_and_type_imports() -> None:
    source = (
        "from chrys.foundation.i18n import MessageDef, MessageRef, msg\n"
        "\n"
        "MESSAGE: MessageDef = msg('dialog.close', fallback='Close')\n"
        "\n"
        "def render(reference: MessageRef) -> MessageRef:\n"
        "    return reference\n"
    )
    _assert_i18n_message_construction_is_canonical({Path("src/chrys/app/good.py"): source})


def test_i18n_message_guard_accepts_fully_qualified_member_annotations() -> None:
    # Deep imports record prefix MODULES as banned values, but terminal
    # member references through them are ordinary qualified type names.
    source = (
        "import chrys.foundation.i18n.messages\n"
        "\n"
        "def render(definition: chrys.foundation.i18n.messages.MessageDef) -> None:\n"
        "    pass\n"
        "\n"
        "def resolve(reference: chrys.foundation.i18n.MessageRef) -> None:\n"
        "    pass\n"
    )
    _assert_i18n_message_construction_is_canonical({Path("src/chrys/app/good.py"): source})


def test_i18n_message_guard_accepts_unrelated_msg_usage_without_import() -> None:
    # ``msg`` stays a legal parameter and attribute name in files that never
    # import the constructor; only calls and shadowing definitions are
    # reserved, or the sweep would flag half the transcript helpers.
    source = (
        "def opens_turn(msg):\n    return msg.role == 'user'\n\ndef label(event):\n    return getattr(event, 'msg')\n"
    )
    _assert_i18n_message_construction_is_canonical({Path("src/chrys/app/good.py"): source})


@pytest.mark.parametrize(
    ("source", "match"),
    [
        (
            "from chrys.foundation.i18n import msg\ndef build():\n    local = msg('dialog.close', fallback='Close')\n",
            "module-level assignment",
        ),
        (
            "from chrys.foundation.i18n import msg\nif enabled:\n    MESSAGE = msg('dialog.close', fallback='Close')\n",
            "module-level assignment",
        ),
        (
            "from chrys.foundation.i18n import msg\nREFERENCE = msg('dialog.close', fallback='Close').bind()\n",
            "module-level assignment",
        ),
        (
            (
                "from chrys.foundation.i18n import msg as translated\n"
                "MESSAGE = translated('dialog.close', fallback='Close')\n"
            ),
            "import msg only",
        ),
        (
            "import chrys.foundation.i18n as i18n\nMESSAGE = i18n.msg('dialog.close', fallback='Close')\n",
            "bare name",
        ),
        (
            "from ..foundation import i18n\nMESSAGE = i18n.msg('dialog.close', fallback='Close')\n",
            "bare name",
        ),
        (
            "from ..foundation.i18n import msg\nMESSAGE = msg('dialog.close', fallback='Close')\n",
            "import msg only",
        ),
        (
            "from chrys.foundation.i18n import msg\nfactory = msg\n",
            "not a first-class value",
        ),
        (
            "from chrys.foundation.i18n import msg\nconsume(msg)\n",
            "not a first-class value",
        ),
        (
            "from chrys.foundation.i18n import msg\ndef factory():\n    return msg\n",
            "not a first-class value",
        ),
        (
            "from chrys.foundation.i18n import msg\ndef factory():\n    return msg('dialog.close', fallback='Close')\n",
            "module-level assignment",
        ),
        (
            (
                "from chrys.foundation.i18n import MessageDef\n"
                "MESSAGE = MessageDef(key='dialog.close', fallback='Close')\n"
            ),
            "construct messages only",
        ),
        (
            "from chrys.foundation.i18n import MessageRef\nREFERENCE = MessageRef(definition=definition)\n",
            "construct messages only",
        ),
        (
            (
                "import dataclasses\n"
                "from chrys.foundation.i18n import msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
                "VARIANT = dataclasses.replace(MESSAGE, fallback='Variant')\n"
            ),
            "dataclasses.replace",
        ),
        (
            (
                "def msg(key, fallback=None):\n"
                "    return (key, fallback)\n"
                "\n"
                "MESSAGE = msg('rogue.key', fallback='Rogue')\n"
            ),
            "locally defined",
        ),
        (
            "MESSAGE = msg('rogue.key', fallback='Rogue')\n",
            "canonical import",
        ),
        (
            "from chrys.foundation import i18n\nMESSAGE = getattr(i18n, 'msg')('dialog.close', fallback='Close')\n",
            "never via getattr",
        ),
        (
            "from chrys.foundation.i18n import messages\nMESSAGE = messages.msg('dialog.close', fallback='Close')\n",
            "bare name",
        ),
        (
            "from ..foundation.i18n import messages\nMESSAGE = messages.msg('dialog.close', fallback='Close')\n",
            "bare name",
        ),
        (
            (
                "import chrys.foundation.i18n.messages\n"
                "MESSAGE = chrys.foundation.i18n.messages.msg('dialog.close', fallback='Close')\n"
            ),
            "bare name",
        ),
        (
            (
                "from chrys.foundation import i18n\n"
                "factory = i18n.msg\n"
                "MESSAGE = factory('dialog.close', fallback='Close')\n"
            ),
            "bare name",
        ),
        (
            "if enabled:\n    from chrys.foundation.i18n import msg\nMESSAGE = msg('dialog.close', fallback='Close')\n",
            "import msg only",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "from rogue_module import msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
            ),
            "rebound",
        ),
        (
            (
                "from chrys.foundation import i18n\n"
                "translated = i18n\n"
                "MESSAGE = translated.msg('dialog.close', fallback='Close')\n"
            ),
            "aliased or passed",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "class msg:\n"
                "    pass\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
            ),
            "locally defined",
        ),
        (
            (
                "import chrys.foundation.i18n\n"
                "translated = chrys.foundation.i18n\n"
                "MESSAGE = translated.msg('dialog.close', fallback='Close')\n"
            ),
            "aliased or passed",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
                "def handle(value):\n"
                "    match value:\n"
                "        case [*msg]:\n"
                "            pass\n"
            ),
            "locally defined",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
                "try:\n"
                "    pass\n"
                "except RuntimeError as msg:\n"
                "    pass\n"
            ),
            "locally defined",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "from chrys.foundation.i18n import MessageDef as msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
            ),
            "rebound",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "from chrys.foundation import i18n as msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
            ),
            "rebound",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "import chrys.foundation.i18n as msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
            ),
            "rebound",
        ),
        (
            (
                "import chrys.foundation.i18n.messages\n"
                "translated = chrys.foundation.i18n\n"
                "MESSAGE = translated.msg('dialog.close', fallback='Close')\n"
            ),
            "aliased or passed",
        ),
        (
            "MESSAGE = msg('dialog.close', fallback='Close')\nfrom chrys.foundation.i18n import msg\n",
            "follow the canonical import",
        ),
        (
            "MESSAGE = msg('dialog.close', fallback='Close'); from chrys.foundation.i18n import msg\n",
            "follow the canonical import",
        ),
        (
            "from chrys.foundation import i18n\nMESSAGE = i18n.messages.msg('dialog.hidden', fallback='Hidden')\n",
            "bare name",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "import msg.submodule\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
            ),
            "rebound",
        ),
        (
            (
                "from dataclasses import replace as clone\n"
                "from chrys.foundation.i18n import msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
                "VARIANT = clone(MESSAGE, fallback='Variant')\n"
            ),
            "dataclasses.replace",
        ),
        (
            (
                "import dataclasses as dc\n"
                "from chrys.foundation.i18n import msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
                "VARIANT = dc.replace(MESSAGE, fallback='Variant')\n"
            ),
            "dataclasses.replace",
        ),
    ],
    ids=[
        "function-local",
        "conditional",
        "inline-bind",
        "aliased-import",
        "module-qualified",
        "relative-module-qualified",
        "relative-import",
        "rebound",
        "passed",
        "returned",
        "wrapper",
        "direct-definition",
        "direct-reference",
        "dataclass-replace",
        "local-def-shadow",
        "unimported-call",
        "getattr-indirection",
        "submodule-qualified",
        "relative-submodule-qualified",
        "plain-import-submodule",
        "aliased-attribute",
        "conditional-canonical-import",
        "rogue-import-rebind",
        "module-alias-first-class",
        "class-shadow",
        "module-alias-dotted",
        "match-star-capture",
        "except-handler-shadow",
        "member-import-rebind",
        "module-member-import-rebind",
        "plain-import-rebind",
        "deep-import-prefix-launder",
        "use-before-import",
        "same-line-use-before-import",
        "nested-submodule-qualified",
        "dotted-plain-import-root-rebind",
        "replace-function-alias",
        "replace-module-alias",
    ],
)
def test_i18n_message_guard_rejects_noncanonical_shapes(source: str, match: str) -> None:
    with pytest.raises(AssertionError, match=match):
        _assert_i18n_message_construction_is_canonical({Path("src/chrys/app/bad.py"): source})
