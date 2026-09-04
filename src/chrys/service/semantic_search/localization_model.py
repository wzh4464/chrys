# Copyright (c) 2026 Chrys. All rights reserved.

"""Run the localization search loop inside the Chrys process.

The skill scripts used to open their own HTTP connection to a model, reading a
model-profile YAML and an API key of their own. That put a second, unmanaged
LLM client in the product: it ignored ``CHRYS_MODEL_LOCK``, it could not reuse
a session's client, and its cost never appeared in any usage accounting.

Here the five read-only search tools are ordinary Chrys tools and the DFS/BFS
loop is Chrys's own agent loop, so localization obeys the same model policy as
everything else. The deterministic stages -- indexing, graph normalization,
CodeGraph, fallback ranking, report rendering -- stay in the subprocess scripts
where they belong.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from chrys.kernel import Agent, FunctionTool
from chrys.service.llm.clients import create_client
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.semantic_search.config import SemanticSearchConfig

logger = logging.getLogger(__name__)

_SKILL_SCRIPT_DIR = Path(__file__).resolve().parent / "skill" / "scripts"

SYSTEM_PROMPT = """You are a specialized repository code-localization agent.

Given an implementation requirement, identify the existing source locations a
developer should inspect or edit. You MUST use the repository tools before
finishing.

Search workflow:
1. Classify the requirement and extract explicit or implicit entry points.
2. Use DFS to follow relevant definitions, child units, callees, and semantic edges.
3. Use BFS to expand to callers, sibling behavior, configuration,
   generated/build surfaces, and validation files when they may require
   coordinated work.
4. Refine the search terms when retrieved code reveals new triggers, state, aliases, or dependency links.
5. Call finish_search only after checking primary logic and plausible propagation paths.

Locations are inspection candidates, not automatic edit mandates. Prefer
functions/methods over entire files. Do not generate a patch and do not access
benchmark answer-side material.

When you are done, reply with ONLY a JSON array of objects:
[{"file": "...", "symbol": "...", "start_line": 1, "end_line": 20, "role": "primary", "reason": "..."}]
"""


@dataclass(frozen=True, slots=True)
class LocalizationRun:
    """What one in-process localization pass produced."""

    locations: list[dict[str, Any]]
    observed_candidates: list[dict[str, Any]]
    tool_call_count: int
    model: str


def _skill_modules() -> tuple[Any, Any, Any]:
    """Import the deterministic graph/tool/parsing modules from the packaged skill.

    They are written as flat scripts that import each other by bare name, which
    is the layout they run under as subprocesses. Putting their directory on
    ``sys.path`` keeps one layout rather than maintaining two.
    """
    directory = str(_SKILL_SCRIPT_DIR)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import _localization_agent  # ty: ignore[unresolved-import]
    import _localization_graph  # ty: ignore[unresolved-import]
    import _localization_tools  # ty: ignore[unresolved-import]

    return _localization_graph, _localization_tools, _localization_agent


def resolve_localization_model_profile(
    settings: Any, registry: Any, active: ModelProfile | None
) -> ModelProfile | None:
    """Prefer the configured cheap profile, else the session's active model.

    Localization is a bounded search over a pre-built graph, so it does not
    need the session's main model; naming a cheaper one here is the whole point
    of the setting.
    """
    configured = settings.semantic_search_model_profile.strip()
    if configured:
        from chrys.service.profiles.models.resolver import resolve_profile_selector

        resolved = resolve_profile_selector(registry, configured)
        if resolved is not None:
            return resolved
        logger.warning("semantic_search.model_profile %r not found; using the active model", configured)
    return active


class ChrysLocalizationModel:
    """Drive the localization search with a Chrys-managed model client."""

    def __init__(
        self,
        profile: ModelProfile,
        *,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        session_dir: Path | None = None,
        client: Any | None = None,
        on_trace: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._profile = profile
        self._session_id = session_id
        self._parent_session_id = parent_session_id
        self._session_dir = session_dir
        self._client = client
        self._on_trace = on_trace

    async def localize(
        self,
        requirement: str,
        *,
        repo: Path,
        index_path: Path,
        codegraph_path: Path | None,
        config: SemanticSearchConfig,
    ) -> LocalizationRun | None:
        """Return ranked locations, or ``None`` when the model produced none."""
        graph_module, tools_module, agent_module = _skill_modules()
        graph = graph_module.LocalizationGraph(
            repo, _read_json(index_path), _read_json(codegraph_path) if codegraph_path else None
        )
        tools = tools_module.LocalizationTools(graph, max_results=config.max_tool_results)
        observed: list[dict[str, Any]] = []
        calls = 0

        def _record(name: str, arguments: dict[str, Any]) -> str:
            nonlocal calls
            calls += 1
            result = tools.execute(name, arguments)
            observed.extend(result.locations)
            self._trace("tool-result", {"name": name, "candidates": len(result.locations)})
            return result.content

        agent = Agent(
            client=self._agent_client(config),
            name="ChrysCodeLocalizer",
            instructions=SYSTEM_PROMPT + f"\n\nRepository: {repo.name}",
            tools=_build_tools(_record),
        )
        self._trace("agent-start", {"model": self._profile.id, "graph": graph.graph_summary()})
        response = await agent.run(
            f"Original requirement:\n{requirement}\n\n"
            "Analyze the issue, identify entry points, and begin repository search.",
            stream=False,
        )
        text = response.text or ""
        # The text itself, bounded: a count alone made a zero-location run
        # undiagnosable after the fact.
        self._trace("final-response", {"chars": len(text), "text": text[:2000]})

        parsed = agent_module.parse_locations(text)
        normalized = agent_module.normalize_locations(parsed, graph, source="llm-search")
        observed_files = {
            str(item.get("file_path") or item.get("file") or "") for item in observed if isinstance(item, dict)
        }
        # A location the search never actually visited is a guess, not a
        # finding; the tool observations are the evidence.
        normalized = [item for item in normalized if item["file_path"] in observed_files]
        if not normalized and observed:
            normalized = agent_module.normalize_locations(
                agent_module.stable_unique(observed), graph, source="tool-observation"
            )
        self._trace("agent-complete", {"locations": len(normalized), "tool_calls": calls})
        if not normalized:
            return None
        return LocalizationRun(
            locations=normalized,
            observed_candidates=agent_module.stable_unique(observed),
            tool_call_count=calls,
            model=self._profile.id,
        )

    def _agent_client(self, config: SemanticSearchConfig) -> Any:
        """Reuse an injected client, else create one bounded to this search."""
        client = self._client
        if client is None:
            client = create_client(
                self._profile,
                session_id=self._session_id,
                parent_session_id=self._parent_session_id,
                session_dir=self._session_dir,
                use_route_session_context=True,
            )
            self._client = client
        client.max_iterations = config.max_iterations
        client.max_function_calls = config.max_iterations * 2
        return client

    def _trace(self, event: str, data: dict[str, Any]) -> None:
        if self._on_trace is not None:
            self._on_trace(event, data)


def _build_tools(execute: Callable[[str, dict[str, Any]], str]) -> list[FunctionTool]:
    """Wrap the five read-only search tools for the Chrys agent loop."""

    def find_file(
        file_name: Annotated[str, "File name or path fragment to look for"],
        dir_path: Annotated[str, "Directory to search under; '.' for the repository root"] = ".",
    ) -> str:
        """Find repository files whose path matches a name or fragment."""
        return execute("find_file", {"file_name": file_name, "dir_path": dir_path})

    def find_code_definition(
        definition_name: Annotated[str, "Class, function, or method name"],
        file_path: Annotated[str, "Optional file to restrict the search to"] = "",
    ) -> str:
        """Find where a class, function, or method is defined."""
        return execute(
            "find_code_definition",
            {"definition_name": definition_name, "file_path": file_path or None},
        )

    def find_code_content(
        query: Annotated[str, "Literal text or regular expression to search for"],
        file_path: Annotated[str, "Optional file to restrict the search to"] = "",
    ) -> str:
        """Search source content for a string or pattern."""
        return execute("find_code_content", {"query": query, "file_path": file_path or None})

    def find_child_unit(
        definition_name: Annotated[str, "Enclosing class or function name"],
        file_path: Annotated[str, "File that declares it"],
    ) -> str:
        """List the methods or nested units declared inside a definition."""
        return execute("find_child_unit", {"definition_name": definition_name, "file_path": file_path})

    def finish_search(
        summary: Annotated[str, "Why the search is complete"] = "",
    ) -> str:
        """Declare the search complete once the primary and propagation paths are covered."""
        return execute("finish_search", {"summary": summary})

    return [
        FunctionTool(func=find_file, name="find_file"),
        FunctionTool(func=find_code_definition, name="find_code_definition"),
        FunctionTool(func=find_code_content, name="find_code_content"),
        FunctionTool(func=find_child_unit, name="find_child_unit"),
        FunctionTool(func=finish_search, name="finish_search"),
    ]


def _read_json(path: Path) -> dict[str, Any]:
    """Read one deterministic-stage artifact, treating an unreadable one as empty."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
