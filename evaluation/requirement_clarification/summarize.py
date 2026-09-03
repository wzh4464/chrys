# Copyright (c) 2026 Chrys. All rights reserved.

"""Create score summaries and strict paired comparisons from Harbor jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.requirement_clarification.protocol import sha256_file, write_json


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """Normalized result for one selected Harbor trial."""

    task: str
    trial: str
    reward: float | None
    error_type: str | None
    patch_sha256: str | None
    clarification_outcome: str | None
    clarification_delta_sha256: str | None
    agent_metadata: dict[str, object]
    input_tokens: int | None
    cache_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    attempts: int


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _task_name(result: dict[str, Any]) -> str:
    task_id = result.get("task_id")
    if isinstance(task_id, dict) and isinstance(task_id.get("path"), str):
        return Path(task_id["path"]).name
    task_name = result.get("task_name")
    if isinstance(task_name, str) and task_name:
        return task_name.rsplit("/", maxsplit=1)[-1]
    raise ValueError("Harbor result has no task identity")


def _error_type(result: dict[str, Any]) -> str | None:
    exception = result.get("exception_info")
    if not isinstance(exception, dict):
        return None
    for key in ("exception_type", "type", "name"):
        value = exception.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _reward(result: dict[str, Any]) -> float | None:
    verifier = result.get("verifier_result")
    if not isinstance(verifier, dict):
        return None
    rewards = verifier.get("rewards")
    if not isinstance(rewards, dict):
        return None
    reward = rewards.get("reward")
    if isinstance(reward, int | float) and not isinstance(reward, bool):
        return float(reward)
    return None


def _agent_metadata(result: dict[str, Any]) -> dict[str, object]:
    agent_result = result.get("agent_result")
    if not isinstance(agent_result, dict):
        return {}
    metadata = agent_result.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _agent_result(result: dict[str, Any]) -> dict[str, Any]:
    agent_result = result.get("agent_result")
    return agent_result if isinstance(agent_result, dict) else {}


def _int_agent_metric(result: dict[str, Any], key: str) -> int | None:
    value = _agent_result(result).get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float_agent_metric(result: dict[str, Any], key: str) -> float | None:
    value = _agent_result(result).get(key)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _clarification_artifacts(trial_dir: Path) -> tuple[str | None, str | None]:
    summaries = sorted(trial_dir.glob("agent/chrys-sessions/**/requirement_clarification/turn_*/summary.json"))
    private_results = sorted(
        trial_dir.glob("agent/chrys-sessions/**/requirement_clarification/turn_*/clarification.private.json")
    )
    workflows = sorted(trial_dir.glob("agent/chrys-sessions/**/requirement_clarification/turn_*/workflow.json"))

    outcome: str | None = None
    if summaries:
        summary = _load_mapping(summaries[-1])
        value = summary.get("outcome")
        outcome = value if isinstance(value, str) else None
    elif workflows:
        workflow = _load_mapping(workflows[-1])
        value = workflow.get("phase")
        outcome = value if isinstance(value, str) else None

    delta_sha256: str | None = None
    if private_results:
        private = _load_mapping(private_results[-1])
        delta = private.get("delta")
        if isinstance(delta, str):
            delta_sha256 = hashlib.sha256(delta.encode()).hexdigest()
    return outcome, delta_sha256


def _select_attempt(attempts: list[tuple[Path, dict[str, Any]]]) -> tuple[Path, dict[str, Any]]:
    return max(
        attempts,
        key=lambda item: (
            _error_type(item[1]) is None,
            str(item[1].get("finished_at") or ""),
            item[0].name,
        ),
    )


def load_selected_attempts(job_dir: Path) -> dict[str, tuple[Path, dict[str, Any], int]]:
    """Select the latest successful Harbor attempt for each task."""
    resolved = job_dir.resolve(strict=True)
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for result_path in sorted(resolved.glob("*/result.json")):
        result = _load_mapping(result_path)
        grouped[_task_name(result)].append((result_path.parent, result))
    if not grouped:
        raise ValueError(f"no trial result.json files found in {resolved}")

    selected: dict[str, tuple[Path, dict[str, Any], int]] = {}
    for task, attempts in sorted(grouped.items()):
        trial_dir, result = _select_attempt(attempts)
        selected[task] = (trial_dir, result, len(attempts))
    return selected


def load_job(job_dir: Path) -> dict[str, TrialRecord]:
    """Load one Harbor job, collapsing retries to the latest successful attempt."""
    records: dict[str, TrialRecord] = {}
    for task, (trial_dir, result, attempt_count) in load_selected_attempts(job_dir).items():
        patch = trial_dir / "artifacts/model.patch"
        outcome, delta_sha256 = _clarification_artifacts(trial_dir)
        records[task] = TrialRecord(
            task=task,
            trial=str(result.get("trial_name") or trial_dir.name),
            reward=_reward(result),
            error_type=_error_type(result),
            patch_sha256=sha256_file(patch) if patch.is_file() else None,
            clarification_outcome=outcome,
            clarification_delta_sha256=delta_sha256,
            agent_metadata=_agent_metadata(result),
            input_tokens=_int_agent_metric(result, "n_input_tokens"),
            cache_tokens=_int_agent_metric(result, "n_cache_tokens"),
            output_tokens=_int_agent_metric(result, "n_output_tokens"),
            cost_usd=_float_agent_metric(result, "cost_usd"),
            attempts=attempt_count,
        )
    return records


def _record_dict(record: TrialRecord) -> dict[str, object]:
    return {
        "task": record.task,
        "trial": record.trial,
        "reward": record.reward,
        "error_type": record.error_type,
        "patch_sha256": record.patch_sha256,
        "clarification_outcome": record.clarification_outcome,
        "clarification_delta_sha256": record.clarification_delta_sha256,
        "agent_metadata": record.agent_metadata,
        "input_tokens": record.input_tokens,
        "cache_tokens": record.cache_tokens,
        "output_tokens": record.output_tokens,
        "cost_usd": record.cost_usd,
        "attempts": record.attempts,
    }


def summarize_job(job_dir: Path) -> dict[str, object]:
    """Build aggregate and per-task views for one job."""
    records = load_job(job_dir)
    scored = [record for record in records.values() if record.reward is not None]
    return {
        "job_dir": str(job_dir.resolve()),
        "task_count": len(records),
        "scored_count": len(scored),
        "error_count": sum(record.error_type is not None for record in records.values()),
        "solved_count": sum(record.reward == 1.0 for record in scored),
        "mean_reward": sum(record.reward or 0.0 for record in scored) / len(scored) if scored else None,
        "tasks": {task: _record_dict(record) for task, record in records.items()},
    }


def _mcnemar_exact(gains: int, regressions: int) -> float:
    discordant = gains + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, regressions) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def compare_jobs(control_dir: Path, candidate_dir: Path) -> dict[str, object]:
    """Compare exactly matching task sets and report paired flips."""
    control = load_job(control_dir)
    candidate = load_job(candidate_dir)
    if control.keys() != candidate.keys():
        missing_candidate = sorted(control.keys() - candidate.keys())
        missing_control = sorted(candidate.keys() - control.keys())
        raise ValueError(
            f"paired task sets differ; missing candidate={missing_candidate}, missing control={missing_control}"
        )

    gains: list[str] = []
    regressions: list[str] = []
    unchanged_solved: list[str] = []
    unchanged_unsolved: list[str] = []
    unscored: list[str] = []
    for task in control:
        left = control[task].reward
        right = candidate[task].reward
        if left is None or right is None:
            unscored.append(task)
        elif left != 1.0 and right == 1.0:
            gains.append(task)
        elif left == 1.0 and right != 1.0:
            regressions.append(task)
        elif left == 1.0:
            unchanged_solved.append(task)
        else:
            unchanged_unsolved.append(task)

    return {
        "schema_version": 1,
        "protocol": "chrys-deepswe-requirement-clarification-v1",
        "control_job": str(control_dir.resolve()),
        "candidate_job": str(candidate_dir.resolve()),
        "task_count": len(control),
        "control_solved": sum(record.reward == 1.0 for record in control.values()),
        "candidate_solved": sum(record.reward == 1.0 for record in candidate.values()),
        "net_solved_delta": len(gains) - len(regressions),
        "gains": gains,
        "regressions": regressions,
        "unchanged_solved": unchanged_solved,
        "unchanged_unsolved": unchanged_unsolved,
        "unscored": unscored,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(len(gains), len(regressions)),
        "control": {task: _record_dict(record) for task, record in control.items()},
        "candidate": {task: _record_dict(record) for task, record in candidate.items()},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, help="Summarize one Harbor job")
    parser.add_argument("--control", type=Path, help="Control Harbor job")
    parser.add_argument("--candidate", type=Path, help="Clarification Harbor job")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run summary or comparison mode."""
    args = _parser().parse_args(argv)
    if args.job is not None and args.control is None and args.candidate is None:
        value = summarize_job(args.job)
    elif args.job is None and args.control is not None and args.candidate is not None:
        value = compare_jobs(args.control, args.candidate)
    else:
        raise ValueError("pass either --job, or both --control and --candidate")
    write_json(args.output, value)
    sys.stdout.write(f"Wrote {args.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
