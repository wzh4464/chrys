# Copyright (c) 2026 Chrys. All rights reserved.

"""LLM-driven DFS/BFS localization loop adapted from SemLoc for Chrys."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _common import ScriptError, now_iso, stable_unique, write_text
from _localization_graph import LocalizationGraph, normalize_relative_path
from _localization_tools import LocalizationTools
from augment_requirement import load_model_profile, load_profile_headers, resolve_env_templates_simple

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
"""

FINAL_PROMPT = """Return the final ranked localization as JSON only. Do not call more tools.

Schema:
{
  "locations": [
    {
      "file_path": "relative/path.py",
      "class_name": "ClassName or empty",
      "function_name": "function_or_method or empty",
      "start_line": 1,
      "end_line": 10,
      "role": "primary | propagation | validation",
      "reason": "why this location is relevant",
      "confidence": "high | medium | low"
    }
  ]
}

Rank highest-confidence primary edit sites first, then propagation/configuration
sites, then validation locations. Include only repository paths observed through
the tools.
"""

_MAX_HTTP_RESPONSE_BYTES = 8 * 1024 * 1024


@dataclass
class AgentRunResult:
    locations: list[dict[str, Any]]
    observed_candidates: list[dict[str, Any]]
    tool_call_count: int
    iteration_count: int
    model: str
    finished: bool


class TraceWriter:
    def __init__(self, path: Path):
        self.path = path
        write_text(self.path, "")

    def write(self, event: str, **data: Any) -> None:
        payload = {"created_at": now_iso(), "event": event, **data}
        flags = os.O_WRONLY | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags)
        with os.fdopen(descriptor, "a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


class OpenAIChatClient:
    """Minimal OpenAI-compatible tool-calling client with deterministic mocks."""

    def __init__(self, model_profile: str, timeout: float, temperature: float):
        self.timeout = timeout
        self.temperature = temperature
        self.mock_responses = self._load_mock_responses()
        if self.mock_responses:
            self.model = "mock-localization"
            self.url = ""
            self.headers: dict[str, str] = {}
            return
        profile = load_model_profile(model_profile)
        provider = str(profile.get("provider", "openai")).lower()
        if provider not in {"openai", "deepseek-openai"}:
            raise ScriptError(f"semantic-search localization requires an OpenAI-compatible profile, got {provider!r}")
        self.model = str(profile.get("model_id", "")).strip()
        if not self.model:
            raise ScriptError("localization model profile does not define model_id")
        base_url = str(profile.get("base_url", "")).strip()
        if not base_url:
            base_url = "https://api.openai.com/v1" if provider == "openai" else "https://api.deepseek.com"
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        if urllib.parse.urlparse(self.url).scheme not in {"http", "https"}:
            raise ScriptError(f"localization model profile uses an unsupported URL scheme: {self.url}")
        api_key = resolve_env_templates_simple(str(profile.get("api_key", "")), location="model profile api_key")
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.headers.update(load_profile_headers(profile))

    @staticmethod
    def _load_mock_responses() -> list[dict[str, Any]]:
        raw = os.environ.get("SEMANTIC_SEARCH_LOCALIZATION_MOCK_RESPONSES", "").strip()
        if not raw:
            single = os.environ.get("SEMANTIC_SEARCH_LOCALIZATION_MOCK_RESPONSE", "").strip()
            if single:
                return [{"role": "assistant", "content": single}]
            return []
        try:
            payload = json.loads(raw)
        except ValueError as err:
            raise ScriptError(f"invalid localization mock responses JSON: {err}") from err
        if isinstance(payload, dict):
            payload = payload.get("responses", [payload])
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ScriptError("localization mock responses must be a JSON array of assistant messages")
        return list(payload)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self.mock_responses:
            return self._normalize_message(self.mock_responses.pop(0))
        if self.model == "mock-localization":
            raise ScriptError("localization mock response sequence was exhausted")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload.update({"tools": tools, "tool_choice": "auto"})
        max_tokens = os.environ.get("SEMANTIC_SEARCH_LOCALIZATION_MAX_TOKENS", "").strip()
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_bytes = response.read(_MAX_HTTP_RESPONSE_BYTES + 1)
                if len(response_bytes) > _MAX_HTTP_RESPONSE_BYTES:
                    raise ScriptError("localization LLM response exceeds 8 MiB")
                raw = response_bytes.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as err:
            detail = err.read(1201).decode("utf-8", errors="replace")[:1200]
            raise ScriptError(f"localization LLM HTTP {err.code}: {detail}") from err
        except urllib.error.URLError as err:
            raise ScriptError(f"localization LLM request failed: {err.reason}") from err
        except TimeoutError as err:
            raise ScriptError(f"localization LLM timed out after {self.timeout:g}s") from err
        try:
            data = json.loads(raw)
            choices = data.get("choices", [])
            message = choices[0].get("message", {})
        except (ValueError, IndexError, AttributeError) as err:
            raise ScriptError(f"invalid localization LLM response: {raw[:1200]}") from err
        return self._normalize_message(message)

    @staticmethod
    def _normalize_message(raw: dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw.get("message"), dict):
            raw = raw["message"]
        message: dict[str, Any] = {"role": "assistant", "content": raw.get("content")}
        tool_calls = []
        for position, item in enumerate(raw.get("tool_calls") or []):
            function = item.get("function", {}) if isinstance(item, dict) else {}
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append(
                {
                    "id": str(item.get("id") or f"mock-call-{position + 1}"),
                    "type": "function",
                    "function": {"name": str(function.get("name", "")), "arguments": str(arguments)},
                }
            )
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message


class LocalizationAgent:
    def __init__(
        self,
        graph: LocalizationGraph,
        tools: LocalizationTools,
        client: OpenAIChatClient,
        trace: TraceWriter,
        *,
        max_iterations: int = 20,
        min_tool_calls: int = 2,
    ):
        self.graph = graph
        self.tools = tools
        self.client = client
        self.trace = trace
        self.max_iterations = max(max_iterations, 1)
        self.min_tool_calls = max(min_tool_calls, 1)

    def run(self, requirement: str) -> AgentRunResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Repository: {self.graph.repo.name}\n\nOriginal requirement:\n{requirement}\n\n"
                    "Analyze the issue, identify entry points, and begin repository search."
                ),
            },
        ]
        observed: list[dict[str, Any]] = []
        final_raw: list[dict[str, Any]] = []
        tool_call_count = 0
        no_tool_retries = 0
        finished = False
        iterations = 0
        self.trace.write("agent-start", model=self.client.model, graph=self.graph.graph_summary())

        for iteration in range(1, self.max_iterations + 1):
            iterations = iteration
            message = self._complete_with_retry(messages, tools=self.tools.definitions)
            messages.append(message)
            self.trace.write(
                "assistant",
                iteration=iteration,
                content=str(message.get("content") or "")[:6000],
                tool_call_count=len(message.get("tool_calls") or []),
            )
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                parsed = parse_locations(message.get("content"))
                if parsed and tool_call_count >= 1:
                    final_raw = parsed
                    break
                if tool_call_count < self.min_tool_calls and no_tool_retries < 3:
                    no_tool_retries += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Use the repository tools before concluding. Follow at least one entry point "
                                "with DFS and check a propagation or validation path with BFS."
                            ),
                        }
                    )
                    self.trace.write("no-tool-nudge", iteration=iteration, retry=no_tool_retries)
                    continue
                break

            no_tool_retries = 0
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = str(function.get("name", ""))
                raw_arguments = str(function.get("arguments", "{}"))
                try:
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments root is not an object")
                except ValueError as err:
                    arguments = {}
                    result_text = f"Invalid tool arguments for {name}: {err}"
                    result_locations: list[dict[str, Any]] = []
                    result_finished = False
                else:
                    result = self.tools.execute(name, arguments)
                    result_text = result.content
                    result_locations = result.locations
                    result_finished = result.finished
                    if name == "finish_search" and tool_call_count < self.min_tool_calls:
                        result_text = (
                            "finish_search was requested too early. Use repository tools to complete DFS/BFS "
                            "coverage before finishing."
                        )
                        result_finished = False
                tool_call_count += 1
                observed.extend(result_locations)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call.get("id", "")),
                        "content": result_text,
                    }
                )
                self.trace.write("tool-call", iteration=iteration, name=name, arguments=arguments)
                self.trace.write(
                    "tool-result",
                    iteration=iteration,
                    name=name,
                    content=result_text[:8000],
                    candidate_count=len(result_locations),
                )
                if result_finished:
                    finished = True
            if finished:
                break

        if not final_raw:
            messages.append({"role": "user", "content": FINAL_PROMPT})
            final_message = self._complete_with_retry(messages, tools=None)
            final_raw = parse_locations(final_message.get("content"))
            self.trace.write("final-response", content=str(final_message.get("content") or "")[:12000])

        normalized = normalize_locations(final_raw, self.graph, source="llm-search")
        observed_files = {
            str(item.get("file_path") or item.get("file") or "") for item in observed if isinstance(item, dict)
        }
        normalized = [item for item in normalized if item["file_path"] in observed_files]
        if not normalized:
            normalized = normalize_locations(stable_unique(observed), self.graph, source="tool-observation")
        self.trace.write(
            "agent-complete",
            location_count=len(normalized),
            tool_call_count=tool_call_count,
            iteration_count=iterations,
            finished=finished,
        )
        return AgentRunResult(
            locations=normalized,
            observed_candidates=stable_unique(observed),
            tool_call_count=tool_call_count,
            iteration_count=iterations,
            model=self.client.model,
            finished=finished,
        )

    def _complete_with_retry(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        last_error: ScriptError | None = None
        for attempt in range(1, 4):
            try:
                return self.client.complete(messages, tools=tools)
            except ScriptError as err:
                last_error = err
                self.trace.write("llm-error", attempt=attempt, error=str(err))
                if attempt < 3 and self.client.model != "mock-localization":
                    time.sleep(attempt)
        raise last_error or ScriptError("localization LLM failed")


def parse_locations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    payload: Any = None
    try:
        payload = json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except ValueError:
                payload = None
    if isinstance(payload, dict):
        locations = payload.get("locations") or payload.get("pred_locations") or []
        return [item for item in locations if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return parse_semloc_text(text)


def parse_semloc_text(text: str) -> list[dict[str, Any]]:
    blocks = re.findall(r"<code_location>(.*?)</code_location>", text, flags=re.DOTALL) or [text]
    locations: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    fields = {
        "file path": "file_path",
        "class name": "class_name",
        "function name": "function_name",
        "start line": "start_line",
        "end line": "end_line",
        "reason": "reason",
        "role": "role",
        "confidence": "confidence",
    }
    for block in blocks:
        for raw_line in block.splitlines():
            line = re.sub(r"^[\s\-*\d.)]+", "", raw_line).replace("**", "").strip()
            if ":" not in line:
                continue
            label, raw_value = line.split(":", 1)
            key = fields.get(label.strip().lower())
            if not key:
                continue
            if key == "file_path" and current:
                locations.append(current)
                current = {}
            cleaned = raw_value.strip().strip("`'\"")
            if key in {"start_line", "end_line"}:
                try:
                    current[key] = int(cleaned)
                except ValueError:
                    continue
            else:
                current[key] = cleaned
        if current:
            locations.append(current)
            current = {}
    return locations


def normalize_locations(
    locations: list[dict[str, Any]], graph: LocalizationGraph, *, source: str
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in locations:
        relative = normalize_relative_path(
            str(raw.get("file_path") or raw.get("file") or raw.get("path") or ""), graph.repo
        )
        if relative not in graph.files:
            continue
        class_name = str(raw.get("class_name") or raw.get("class") or "").strip()
        function_name = str(
            raw.get("function_name") or raw.get("function") or raw.get("symbol") or raw.get("name") or ""
        ).strip()
        resolved = graph.resolve_unit(relative, function_name or class_name)
        if function_name and not class_name and "." in function_name and not resolved:
            class_name, function_name = function_name.rsplit(".", 1)
            resolved = graph.resolve_unit(relative, function_name or class_name)
        unit = resolved[0] if resolved else None
        if unit:
            class_name = class_name or str(unit.get("class_name") or "")
            if unit.get("kind") == "class" and not function_name:
                class_name = str(unit.get("name") or class_name)
            elif not function_name:
                function_name = str(unit.get("name") or "")
        start_line = _as_int(raw.get("start_line")) or _as_int((unit or {}).get("start_line"))
        end_line = _as_int(raw.get("end_line")) or _as_int((unit or {}).get("end_line")) or start_line
        key = (relative, class_name, function_name, start_line, end_line)
        if key in seen:
            continue
        seen.add(key)
        rank = len(normalized) + 1
        is_test = bool(graph.files[relative].get("is_test"))
        role = str(raw.get("role") or "").lower()
        if role not in {"primary", "propagation", "validation"}:
            role = "validation" if is_test else ("primary" if rank <= 3 else "propagation")
        confidence = str(raw.get("confidence") or "medium").lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        symbol = f"{class_name}.{function_name}" if class_name and function_name else (function_name or class_name)
        normalized.append(
            {
                "rank": rank,
                "role": role,
                "repo": graph.files[relative].get("repo", ""),
                "file": relative,
                "file_path": relative,
                "symbol": symbol,
                "class_name": class_name,
                "function_name": function_name,
                "start_line": start_line,
                "end_line": end_line,
                "reason": str(raw.get("reason") or "Repository search evidence matched this location."),
                "evidence": {"source": source},
                "confidence": confidence,
                "must_verify": True,
            }
        )
    return normalized


def _as_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except TypeError, ValueError:
        return None
    return parsed if parsed > 0 else None
