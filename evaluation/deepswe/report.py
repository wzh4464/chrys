#!/usr/bin/env python3
"""Write the four-section report for a DeepSWE run of chrys on the long-horizon track.

For every task it reads the captured chrys session store (clarification and
localization artifacts, the memory prior, the campaign stream, the route marker), the
patch, and — when present — the Harbor verifier's grade, and renders Markdown:

    1. requirement clarification + code localization
    2. ContextGraph recall + deposit
    3. PACT execution
    4. grade

It understands both layouts: LoLBench runs (``runs/deepswe/<agent>/<id>/agent_out/chrys``)
and the plain DeepSWE runner (``<output-dir>/<task>/result.json`` with the sessions under
``<output-dir>/chrys-home/.chrys/sessions``).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None


def _sessions(store: Path) -> list[Path]:
    if not store.is_dir():
        return []
    return sorted(
        (p for p in store.iterdir() if p.is_dir() and (p / "trajectory").exists()), key=lambda p: p.stat().st_mtime
    )


def _main_session(store: Path) -> Path | None:
    """The session that ran the routed turn: it owns requirement_clarification/."""
    for session in reversed(_sessions(store)):
        if (session / "requirement_clarification").is_dir() or (session / "long_horizon").is_dir():
            return session
    sessions = _sessions(store)
    return sessions[-1] if sessions else None


def _route_marker(session: Path) -> dict | None:
    envelope = _load(session / "session.json") or _load(session / "session.recovery.json")
    if not envelope:
        return None
    for message in envelope.get("state", {}).get("messages", []):
        route = (message.get("additional_properties") or {}).get("_chrys_route")
        if route:
            return route
    return None


def _clarification(session: Path) -> dict:
    turn = session / "requirement_clarification" / "turn_1"
    summary = _load(turn / "05-outcome" / "summary.json") or {}
    private = _load(turn / "clarification.private.json") or {}
    investigations = []
    for path in sorted((turn / "03-clarification" / "investigations").glob("proposal-*.json")):
        data = _load(path) or {}
        inv = data.get("investigation", data)
        investigations.append(
            f"{inv.get('status')}/{inv.get('coverage_status')} calls={len(inv.get('tool_calls', []))} reads={len(inv.get('inspected_paths', []))}"
        )
    points = [g.get("statement", "") for g in (private.get("selection") or {}).get("guidance_points", [])]
    return {
        "outcome": summary.get("outcome"),
        "accepted_phase": summary.get("accepted_phase"),
        "status": private.get("status") or summary.get("clarification_status"),
        "empty_reason": private.get("empty_reason") or summary.get("clarification_empty_reason"),
        "elapsed": private.get("elapsed_seconds"),
        "investigations": investigations,
        "guidance": points,
    }


def _localization(session: Path) -> dict:
    root = session / "long_horizon" / "turn_1" / "semantic-search"
    loc = _load(root / "code-localization.json") or {}
    candidates = loc.get("candidates") or loc.get("locations") or []
    perception = (
        (root / "codegraph-perception.md").read_text(encoding="utf-8", errors="replace")
        if (root / "codegraph-perception.md").is_file()
        else ""
    )
    available = "Available: True" in perception
    return {
        "candidates": [
            f"{c.get('path')}:{c.get('start_line', c.get('line', ''))} [{c.get('role', '')}] {c.get('confidence', '')}"
            for c in candidates[:8]
        ],
        "count": len(candidates),
        "codegraph": "available" if available else ("unavailable" if perception else "not run"),
    }


def _prior(session: Path) -> dict:
    path = session / "long_horizon" / "turn_1" / "memory-prior.md"
    if not path.is_file():
        return {"status": "no memory-prior.md (older tree)", "rules": 0}
    text = path.read_text(encoding="utf-8", errors="replace")
    status = next((line[len("Recall: ") :] for line in text.splitlines() if line.startswith("Recall: ")), "?")
    return {"status": status, "rules": sum(1 for line in text.splitlines() if line.startswith("- "))}


def _text(path: Path, limit: int) -> str:
    try:
        body = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return body if len(body) <= limit else body[:limit].rstrip() + f"\n… ({len(body) - limit} more chars)"


def _deposit(session: Path) -> dict:
    envelope = _load(session / "session.json") or _load(session / "session.recovery.json") or {}
    state = envelope.get("state", {})
    return {"watermark": state.get("memory_deposit_watermark"), "turns": state.get("turn_counter")}


def _campaign(session: Path) -> dict:
    out = {"children": [], "marker": None}
    for path in sorted((session / "sub_agents" / "sessions").glob("chrys_pact_*.json")):
        data = _load(path) or {}
        meta = data.get("meta", {})
        acp = data.get("acp_state", {})
        updates = acp.get("translated_updates", [])
        titles = [u["update"].get("title") for u in updates if u.get("update", {}).get("title")]
        out["children"].append(
            {
                "status": meta.get("status"),
                "stop": acp.get("stop_reason"),
                "started": (meta.get("created_at") or "")[11:19],
                "ended": (meta.get("ended_at") or "")[11:19],
                "updates": len(updates),
                "roles": sorted({t for t in titles if t and t.startswith("PACT ")}),
                "result": (meta.get("result_preview") or "").replace("\n", " / ")[:240],
            }
        )
    return out


def _task_dirs(runs_dir: Path) -> list[tuple[str, Path | None, Path | None]]:
    """(task id, chrys session store, patch) for either layout."""
    found = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir() or entry.name in {"chrys-home", "grades", "workspaces"}:
            continue
        store = entry / "agent_out" / "chrys"
        # LoLBench captures solution.patch; the DeepSWE runner writes model.patch.
        patch = next((c for c in (entry / "solution.patch", entry / "model.patch") if c.is_file()), None)
        if store.is_dir() or patch is not None:
            found.append((entry.name, store if store.is_dir() else None, patch))
        elif (entry / "result.json").is_file():
            found.append((entry.name, None, None))
    return found


def _runner_sessions(runs_dir: Path, task_id: str) -> Path | None:
    """The plain DeepSWE runner keeps one home for every task; match by workspace path."""
    store = runs_dir / "chrys-home" / ".chrys" / "sessions"
    for session in reversed(_sessions(store)):
        # A run that ended without a final response leaves only the recovery envelope.
        envelope = _load(session / "session.json") or _load(session / "session.recovery.json") or {}
        meta = envelope.get("meta", {})
        cwd = str(meta.get("primary_cwd") or meta.get("cwd") or "")
        if cwd.rstrip("/").endswith(task_id) and (
            (session / "requirement_clarification").is_dir() or (session / "long_horizon").is_dir()
        ):
            return session
    return None


def render(runs_dir: Path, grades_dir: Path | None, *, verbose: bool = False) -> str:
    grades: dict[str, dict] = {}
    if grades_dir and (grades_dir / "results.csv").is_file():
        with (grades_dir / "results.csv").open(newline="", encoding="utf-8") as fh:
            grades = {row["instance_id"]: row for row in csv.DictReader(fh)}
    lines = ["# DeepSWE × chrys long-horizon — run report", "", f"Run directory: `{runs_dir}`", ""]
    totals = {
        "tasks": 0,
        "clarified": 0,
        "degraded": 0,
        "prior": 0,
        "campaigns": 0,
        "campaign_completed": 0,
        "resolved": 0,
        "graded": 0,
        "patches": 0,
    }
    for task_id, store, patch in _task_dirs(runs_dir):
        session = _main_session(store) if store else _runner_sessions(runs_dir, task_id)
        totals["tasks"] += 1
        lines.append(f"## {task_id}")
        lines.append("")
        if session is None:
            lines.append("_no chrys session captured_")
            lines.append("")
            continue
        route = _route_marker(session) or {}
        clar = _clarification(session)
        loc = _localization(session)
        prior = _prior(session)
        dep = _deposit(session)
        camp = _campaign(session)
        grade = grades.get(task_id)
        totals["clarified"] += 1 if clar["status"] == "completed" else 0
        totals["degraded"] += 1 if clar["status"] == "degraded" else 0
        totals["prior"] += 1 if str(prior["status"]).startswith("recalled") else 0
        totals["campaigns"] += len(camp["children"])
        totals["campaign_completed"] += sum(1 for c in camp["children"] if c["status"] == "completed")
        if patch is not None and patch.stat().st_size > 0:
            totals["patches"] += 1
        if grade:
            totals["graded"] += 1
            totals["resolved"] += 1 if grade.get("resolved") == "True" else 0
        lines.append(
            f"**Route**: track={route.get('track')} source={route.get('source')} baseline={route.get('baseline')} campaign={json.dumps(route.get('campaign'))}"
        )
        lines.append("")
        lines.append("### 1. 需求澄清 + 需求定位")
        lines.append(
            f"- clarification: `{clar['status']}`"
            + (f" ({clar['empty_reason']})" if clar["empty_reason"] else "")
            + f", outcome `{clar['outcome']}` / accepted `{clar['accepted_phase']}`, {float(clar['elapsed'] or 0):.0f}s"
        )
        if clar["investigations"]:
            lines.append("- investigations: " + "; ".join(clar["investigations"]))
        for point in clar["guidance"][:4]:
            lines.append(f"  - {point[:220]}")
        lines.append(f"- localization: {loc['count']} candidates, CodeGraph {loc['codegraph']}")
        for cand in loc["candidates"]:
            lines.append(f"  - {cand}")
        if verbose:
            delta = _text(
                session / "requirement_clarification" / "turn_1" / "05-outcome" / "clarified-requirement-delta.md", 2500
            )
            if delta:
                lines.extend(
                    (
                        "",
                        "<details><summary>澄清增量（clarified-requirement-delta.md）</summary>",
                        "",
                        "```text",
                        delta,
                        "```",
                        "</details>",
                    )
                )
        lines.append("")
        lines.append("### 2. ContextGraph 召回与沉淀")
        lines.append(f"- recall: {prior['status']} ({prior['rules']} items)")
        lines.append(f"- deposit watermark: {dep['watermark']} of {dep['turns']} turn(s)")
        if verbose:
            recalled = _text(session / "long_horizon" / "turn_1" / "memory-prior.md", 2000)
            if recalled:
                lines.extend(
                    (
                        "",
                        "<details><summary>召回内容（memory-prior.md）</summary>",
                        "",
                        "```text",
                        recalled,
                        "```",
                        "</details>",
                    )
                )
        lines.append("")
        lines.append("### 3. PACT 执行")
        if not camp["children"]:
            lines.append("- no campaign (no verify command for this workspace, or delegation not planned)")
        for child in camp["children"]:
            lines.append(
                f"- {child['status']} ({child['stop']}) {child['started']}→{child['ended']}, {child['updates']} updates, roles {', '.join(child['roles']) or '-'}"
            )
            lines.append(f"  - {child['result']}")
        lines.append("")
        if verbose:
            final = _text(session / "requirement_clarification" / "turn_1" / "05-outcome" / "final-response.md", 2500)
            if final:
                lines.extend(("### 最终回复（final-response.md）", "", "```text", final, "```", ""))
        lines.append("### 4. 判分")
        if grade:
            lines.append(
                f"- {grade.get('status')}: reward={grade.get('reward')} f2p={grade.get('f2p_passed')}/{grade.get('f2p_total')} p2p={grade.get('p2p_passed')}/{grade.get('p2p_total')} ({grade.get('elapsed_s')}s)"
            )
        else:
            lines.append(
                f"- patch: {'%d bytes' % patch.stat().st_size if patch is not None and patch.is_file() else 'none'}; not graded"
            )
        lines.append("")
    # "complete" is the whole track: a campaign ran to completion on the clarified plan.
    lines.insert(
        3,
        "| tasks | complete (campaign completed) | campaigns started | clarified | degraded "
        "| prior recalled | patches | graded | resolved |",
    )
    lines.insert(4, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.insert(
        5,
        f"| {totals['tasks']} | {totals['campaign_completed']} | {totals['campaigns']} | {totals['clarified']} "
        f"| {totals['degraded']} | {totals['prior']} | {totals['patches']} | {totals['graded']} | {totals['resolved']} |",
    )
    lines.insert(6, "")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="runs/deepswe/<agent> (LoLBench) or the DeepSWE runner's output dir",
    )
    parser.add_argument("--grades", type=Path, default=None, help="grade.py --out-dir (results.csv)")
    parser.add_argument("--out", type=Path, default=Path("report.md"))
    parser.add_argument(
        "--verbose", action="store_true", help="include each stage's actual output (delta, prior, final response)"
    )
    args = parser.parse_args(argv)
    text = render(args.runs_dir, args.grades, verbose=args.verbose)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
