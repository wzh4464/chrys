# Copyright (c) 2026 Chrys. All rights reserved.

"""Turn a code search's candidates into evidence the next pass can use.

Everything here is bounded and explicitly labelled untrusted. A ranked guess
about where code lives is useful to a repair pass and to a campaign's plan; it
is not an instruction, and it must not be able to grow past the reminder budget
it shares with the clarification it accompanies.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_HINT_ITEMS = 8
MAX_HINT_CHARS = 2000
MAX_REMINDER_CHARS = 3000
MAX_BRIEF_LOCATIONS = 12

_UNTRUSTED_HEADER = "Code localization (untrusted; verify before editing)"


def _rows(locations: Sequence[dict[str, Any]], limit: int) -> list[str]:
    """Render at most *limit* locations as one line each."""
    rows: list[str] = []
    for location in locations[:limit]:
        path = str(location.get("file") or location.get("file_path") or "").strip()
        if not path:
            continue
        symbol = str(location.get("symbol") or location.get("function_name") or "").strip()
        start = location.get("start_line")
        end = location.get("end_line")
        span = f":{start}-{end}" if isinstance(start, int) and isinstance(end, int) and end >= start else ""
        role = str(location.get("role") or "").strip()
        reason = " ".join(str(location.get("reason") or "").split())[:160]
        parts = [f"{path}{span}"]
        if symbol:
            parts.append(symbol)
        if role:
            parts.append(f"[{role}]")
        if reason:
            parts.append(f"- {reason}")
        rows.append("- " + " ".join(parts))
    return rows


def _bounded(rows: Sequence[str], max_chars: int) -> str:
    """Join rows, stopping before the budget rather than truncating mid-row."""
    out: list[str] = []
    used = 0
    for row in rows:
        if used + len(row) + 1 > max_chars:
            break
        out.append(row)
        used += len(row) + 1
    return "\n".join(out)


def localization_hints(
    locations: Sequence[dict[str, Any]],
    *,
    max_items: int = MAX_HINT_ITEMS,
    max_chars: int = MAX_HINT_CHARS,
) -> str:
    """Return candidate locations for a plan prompt, or ``""`` when there are none."""
    return _bounded(_rows(locations, max_items), max_chars)


def augment_delta_with_locations(
    delta_text: str,
    locations: Sequence[dict[str, Any]],
    *,
    max_items: int = MAX_HINT_ITEMS,
    max_chars: int = MAX_REMINDER_CHARS,
) -> str:
    """Append the candidate table after the clarification delta, verbatim.

    The delta comes first and is never edited: it is the repair's authority,
    and the search results are an aid to finding what it describes.
    """
    table = _bounded(_rows(locations, max_items), max_chars)
    if not table:
        return delta_text
    return f"{delta_text}\n\n{_UNTRUSTED_HEADER}\n{table}"


def build_task_brief(
    *,
    original_requirement: str,
    clarified_requirement_md: str | None,
    locations: Sequence[dict[str, Any]],
    baseline: str,
    warnings: Sequence[str] = (),
    max_locations: int = MAX_BRIEF_LOCATIONS,
) -> str:
    """Render the brief a campaign's roles read to understand the task.

    Written even when a stage degraded: a brief that says what is missing is
    more useful to a role than no brief at all.
    """
    sections = [
        "# Task brief",
        "",
        "## Original requirement (authority)",
        original_requirement.strip() or "(none recorded)",
        "",
        "## Clarified requirement",
        (clarified_requirement_md or "").strip() or "(clarification produced none)",
        "",
        f"## Current baseline\nThe workspace holds: {baseline}",
        "",
        f"## {_UNTRUSTED_HEADER}",
        _bounded(_rows(locations, max_locations), MAX_REMINDER_CHARS) or "(no candidate locations)",
    ]
    if warnings:
        sections.extend(["", "## Degraded stages", *[f"- {warning}" for warning in warnings]])
    return "\n".join(sections).rstrip() + "\n"


@dataclass(frozen=True, slots=True)
class PactRunRequest:
    """The one JSON prompt that hands a campaign its accepted inputs."""

    request_id: str
    contract_path: str
    """Workspace-relative, POSIX."""
    plan_path: str

    def to_json(self) -> str:
        """Render the launch contract exactly as ``chrys pact-agent`` parses it."""
        return json.dumps(
            {
                "schema": "chrys-pact/run-request/v1",
                "contract_path": self.contract_path,
                "plan_path": self.plan_path,
            }
        )


def materialize_pact_request(workspace_cwd: Path, pact_input_dir: Path, request_id: str) -> PactRunRequest:
    """Copy the accepted pair into ``.pact-io/`` and describe where they landed.

    The campaign runs in the workspace, so its inputs have to be inside it and
    named relatively: an absolute path from this session's artifact tree would
    not resolve for the agent that reads them.
    """
    destination = workspace_cwd / ".pact-io" / "chrys-pact" / request_id
    destination.mkdir(parents=True, exist_ok=True)
    contract = destination / "goal-contract.json"
    plan = destination / "initial-plan.json"
    contract.write_bytes((pact_input_dir / "goal-contract.json").read_bytes())
    plan.write_bytes((pact_input_dir / "initial-plan.json").read_bytes())
    return PactRunRequest(
        request_id=request_id,
        contract_path=contract.relative_to(workspace_cwd).as_posix(),
        plan_path=plan.relative_to(workspace_cwd).as_posix(),
    )


def build_delegation_reminder(
    *,
    brief_path: Path,
    brief_summary: str,
    baseline: str,
    request: PactRunRequest,
    pact_tool: str,
    max_chars: int = 6000,
) -> str:
    """Tell the model exactly what to send, and what not to do instead."""
    reminder = (
        "[LONG_HORIZON_DELEGATION]\n"
        f"The workspace holds the {baseline} baseline. A governed PACT campaign will verify it and "
        "finish the remaining work.\n\n"
        f"Call `{pact_tool}` exactly once, with this JSON as the entire prompt:\n"
        f"{request.to_json()}\n\n"
        f"The task brief is at {brief_path}. Its summary:\n{brief_summary.strip()}\n\n"
        "Rules:\n"
        "- Report the campaign's returned status verbatim (`completed` / `blocked` / `active`).\n"
        "- Never describe a non-completed campaign as finished.\n"
        "- Do not implement the same work yourself while the campaign runs.\n"
    )
    return reminder[:max_chars]
