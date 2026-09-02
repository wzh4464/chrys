# Copyright (c) 2026 Chrys. All rights reserved.

"""Summarize a DeepSWE/Chrys batch run without reading gold answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Directory produced by deepswe_runner.py")
    parser.add_argument("--dataset", help="Optional DeepSWE checkout used to restore manifest order")
    parser.add_argument(
        "--refresh-summary",
        action="store_true",
        help="Rebuild summary.json from per-task result and verification files",
    )
    parser.add_argument("--output", help="Optional summary output JSON")
    return parser


def _manifest_order(dataset: Path) -> list[str]:
    tasks_root = dataset / "tasks" if (dataset / "tasks" / "manifest.json").is_file() else dataset
    value = json.loads((tasks_root / "manifest.json").read_text(encoding="utf-8"))
    tasks = value.get("tasks", []) if isinstance(value, dict) else value
    return [str(item.get("task_id", "")) for item in tasks if isinstance(item, dict) and item.get("task_id")]


def _refresh_records(run_dir: Path, dataset: Path | None) -> list[dict[str, Any]]:
    order = _manifest_order(dataset) if dataset is not None else []
    positions = {task_id: position for position, task_id in enumerate(order, 1)}
    records: list[dict[str, Any]] = []
    for result_path in run_dir.glob("*/result.json"):
        value = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            continue
        record = dict(value)
        task_id = str(record.get("task_id") or result_path.parent.name)
        record["task_id"] = task_id
        if task_id in positions:
            record["position"] = positions[task_id]
        error = str(record.get("error", ""))
        if record.get("status") == "failed" and "timed out after" in error:
            record["status"] = "agent_timeout"
        patch = result_path.parent / "model.patch"
        record["patch_bytes"] = patch.stat().st_size if patch.is_file() else 0
        verification_path = result_path.parent / "verification.json"
        if verification_path.is_file():
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            if isinstance(verification, dict):
                record["verification_status"] = verification.get("status")
                record["resolved"] = verification.get("resolved") is True
                record["reward"] = verification.get("reward", {})
        records.append(record)
    return sorted(records, key=lambda item: (int(item.get("position", 1_000_000)), str(item.get("task_id", ""))))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    summary_path = run_dir / "summary.json"
    dataset = Path(args.dataset).expanduser().resolve() if args.dataset else None
    if args.refresh_summary:
        records = _refresh_records(run_dir, dataset)
        summary_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif not summary_path.is_file():
        raise SystemExit(f"Run summary does not exist: {summary_path}")
    else:
        records = json.loads(summary_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    durations: list[float] = []
    for record in records:
        status = str(record.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        duration = record.get("duration_seconds")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
    verification_path = run_dir / "verification-summary.json"
    verification: dict[str, Any] = {}
    if verification_path.is_file():
        value = json.loads(verification_path.read_text(encoding="utf-8"))
        verification = value if isinstance(value, dict) else {}
    payload = {
        "format": "chrys-deepswe-evaluation",
        "task_count": len(records),
        "status_counts": counts,
        "localized_count": sum(record.get("localization_returncode") == 0 for record in records),
        "agent_success_count": counts.get("completed", 0),
        "nonempty_patch_count": sum(int(record.get("patch_bytes", 0)) > 0 for record in records),
        "verified_count": verification.get("verified_count", 0),
        "resolved_count": verification.get("resolved_count", 0),
        "resolve_rate": verification.get("resolve_rate"),
        "mean_duration_seconds": round(sum(durations) / len(durations), 3) if durations else 0.0,
        "note": "This is an execution summary; gold patches and tests are evaluated separately offline.",
    }
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
