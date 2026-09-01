# Copyright (c) 2026 Chrys. All rights reserved.

"""Pure, score-free contracts shared by the DeepSWE evaluation tools."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

CONTROL_ARM = "control"
CANDIDATE_ARM = "clarification"
ARMS = (CONTROL_ARM, CANDIDATE_ARM)
REPAIR_ARM = "fixed-p0-repair"
ADAPTER_MODES = (*ARMS, REPAIR_ARM)

MODEL_PROFILE_ID = "deepseek-v4-pro-0813-openrouter"
MODEL_ID = "deepseek/deepseek-v4-pro-0813"
HARBOR_MODEL_NAME = f"openrouter/{MODEL_ID}"
OPENROUTER_HOST = "openrouter.ai"
CODING_PHASE_TIMEOUT_SECONDS = 5400.0

AGENT_IMPORT_PATH = "evaluation.requirement_clarification.harbor_agent:ChrysHarborAgent"
CONTROL_PROFILE_NAME = "DeepSWEControl"
CANDIDATE_PROFILE_NAME = "DeepSWEClarification"
REPAIR_PROFILE_NAME = "DeepSWEFixedP0Repair"

_FIXED_P0_REPAIR_INSTRUCTIONS = """You are Chrys running a bounded fixed-P0 repair experiment.

The workspace already contains the exact matched baseline implementation P0. Do not implement the
requirement from scratch, reset the workspace, create branches, or commit. Preserve every correct P0
change. The task text contains the authoritative original requirement followed by a small ΔR section.
Your only objective is to repair concrete P0 gaps identified by ΔR.

Use this bounded workflow:
1. Inspect `git diff --stat` and the relevant portions of `git diff` once. Map each ΔR bullet to P0.
2. If P0 already satisfies every ΔR bullet, make no changes and finish immediately.
3. Otherwise read only the directly implicated declarations, data flow, consumers, and existing tests.
   Do not re-audit the whole feature, add unrelated hardening, refactor, or improve style.
4. Make the minimum edits needed for unmet ΔR bullets. Add or change tests only when they directly
   verify those edits.
5. Run one focused test command. If it fails, diagnose the concrete failure and repair it; do not try
   many equivalent command variants. After focused tests pass, run at most one broader relevant suite
   when it is reasonably fast. Do not run exhaustive manual end-to-end matrices.
6. Review the final diff once against ΔR and stop. Do not keep exploring after the mapped gaps and
   focused tests are complete.

Keep tool calls purposeful and batched. Prefer one search/read operation that answers a question over
many incremental probes. Report only the retained P0 behavior, the minimal repair, and verification.
"""

_PROFILE_IDS = {
    CONTROL_ARM: "d33e5e000001",
    CANDIDATE_ARM: "d33e5e000002",
}
_PROFILE_NAMES = {
    CONTROL_ARM: CONTROL_PROFILE_NAME,
    CANDIDATE_ARM: CANDIDATE_PROFILE_NAME,
}


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    """A stable identity for one experiment input."""

    path: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class TaskFingerprint:
    """A task identity shared by both paired arms."""

    name: str
    sha256: str
    file_count: int


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: Path) -> FileFingerprint:
    """Build a serializable fingerprint for a required file."""
    resolved = path.resolve(strict=True)
    return FileFingerprint(path=str(resolved), sha256=sha256_file(resolved), size=resolved.stat().st_size)


def fingerprint_task(task_dir: Path) -> TaskFingerprint:
    """Hash every regular task input using relative paths and bytes."""
    digest = hashlib.sha256()
    files = sorted(path for path in task_dir.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(task_dir).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return TaskFingerprint(name=task_dir.name, sha256=digest.hexdigest(), file_count=len(files))


def fingerprint_dataset(dataset_dir: Path, *, expected_tasks: int | None = None) -> list[TaskFingerprint]:
    """Validate and fingerprint a local Harbor dataset."""
    resolved = dataset_dir.resolve(strict=True)
    task_dirs = sorted(path for path in resolved.iterdir() if path.is_dir())
    if expected_tasks is not None and len(task_dirs) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} tasks in {resolved}, found {len(task_dirs)}")
    if not task_dirs:
        raise ValueError(f"no task directories found in {resolved}")

    missing = [path.name for path in task_dirs if not (path / "task.toml").is_file()]
    if missing:
        raise ValueError(f"task.toml missing for: {', '.join(missing)}")
    return [fingerprint_task(path) for path in task_dirs]


def render_paired_agent_profiles(code_profile_path: Path, output_dir: Path) -> dict[str, Path]:
    """Render two profiles that differ only in the clarification flag and identity."""
    source = yaml.safe_load(code_profile_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError(f"agent profile must be a mapping: {code_profile_path}")
    if source.get("name") != "Code":
        raise ValueError(f"expected the built-in Code profile, got {source.get('name')!r}")
    if "requirement_clarification" in source:
        raise ValueError("built-in Code profile unexpectedly sets requirement_clarification")

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, Path] = {}
    for arm in ARMS:
        profile = dict(source)
        profile["name"] = _PROFILE_NAMES[arm]
        profile["id"] = _PROFILE_IDS[arm]
        profile["display_name"] = f"DeepSWE {'Control' if arm == CONTROL_ARM else 'Clarification'}"
        profile["description"] = "Pinned DeepSWE requirement-clarification experiment profile"
        profile["requirement_clarification"] = {
            "enabled": arm == CANDIDATE_ARM,
            "initial_timeout_seconds": CODING_PHASE_TIMEOUT_SECONDS,
            "repair_timeout_seconds": CODING_PHASE_TIMEOUT_SECONDS,
        }
        destination = output_dir / f"{arm}.yaml"
        destination.write_text(yaml.safe_dump(profile, sort_keys=False, width=120), encoding="utf-8")
        rendered[arm] = destination
    assert_paired_profiles(rendered[CONTROL_ARM], rendered[CANDIDATE_ARM])
    return rendered


def render_fixed_p0_repair_profile(code_profile_path: Path, output_path: Path) -> Path:
    """Render a minimal, incrementally scoped agent profile for fixed-P0 repair."""
    source = yaml.safe_load(code_profile_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("name") != "Code":
        raise ValueError(f"expected the built-in Code profile: {code_profile_path}")
    profile = dict(source)
    profile.update(
        {
            "name": REPAIR_PROFILE_NAME,
            "id": "d33e5e000003",
            "display_name": "DeepSWE Fixed-P0 Repair",
            "description": "Bounded incremental repair of a frozen baseline P0",
            "instructions": _FIXED_P0_REPAIR_INSTRUCTIONS,
            "tools": {"builtins": ["filesystem.write", "filesystem.read", "shell", "search"]},
            "sub_agents": {"max_total_concurrency": 1, "agents": []},
            "requirement_clarification": {
                "enabled": False,
                "initial_timeout_seconds": CODING_PHASE_TIMEOUT_SECONDS,
                "repair_timeout_seconds": CODING_PHASE_TIMEOUT_SECONDS,
            },
        }
    )
    for section in ("skills", "compaction", "memory"):
        profile.pop(section, None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(profile, sort_keys=False, width=120), encoding="utf-8")
    return output_path


def assert_paired_profiles(control_path: Path, candidate_path: Path) -> None:
    """Fail unless the two profiles differ only in identity and feature flag."""
    control = yaml.safe_load(control_path.read_text(encoding="utf-8"))
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    for profile in (control, candidate):
        if not isinstance(profile, dict):
            raise ValueError("rendered agent profile must be a mapping")

    ignored = {"name", "id", "display_name", "requirement_clarification"}
    control_shared = {key: value for key, value in control.items() if key not in ignored}
    candidate_shared = {key: value for key, value in candidate.items() if key not in ignored}
    if control_shared != candidate_shared:
        raise ValueError("paired agent profiles differ outside identity and requirement_clarification")
    expected_timeouts = {
        "initial_timeout_seconds": CODING_PHASE_TIMEOUT_SECONDS,
        "repair_timeout_seconds": CODING_PHASE_TIMEOUT_SECONDS,
    }
    if control["requirement_clarification"] != {"enabled": False, **expected_timeouts}:
        raise ValueError("control profile must disable requirement clarification")
    if candidate["requirement_clarification"] != {"enabled": True, **expected_timeouts}:
        raise ValueError("candidate profile must enable requirement clarification")


def expected_model_lock() -> str:
    """Return the only model lock accepted by the evaluation adapter."""
    return json.dumps(
        {
            "provider": "openai",
            "api_style": "chat_completions",
            "model_id": MODEL_ID,
            "base_url": "https://openrouter.ai/api/v1",
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def normalized_model_lock(raw: str) -> str:
    """Normalize a model lock for equality without accepting extra fields."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("CHRYS_MODEL_LOCK must be valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError("CHRYS_MODEL_LOCK must be a JSON object")
    return json.dumps(parsed, separators=(",", ":"), sort_keys=True)


def validate_run_id(value: str) -> str:
    """Accept a portable single-component run label."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError("run id must be one filename-safe component")
    return value


def read_secrets_env(path: Path) -> dict[str, str]:
    """Read the small experiment dotenv file without returning unrelated values."""
    values: dict[str, str] = {}
    assignment = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = assignment.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid dotenv assignment at {path}:{line_number}")
        key, value = match.groups()
        if value[:1] == value[-1:] and value[:1] in {'"', "'"}:
            value = value[1:-1]
        if key in {"OPENROUTER_API_KEY", "CHRYS_MODEL_LOCK"}:
            values[key] = value
    if not values.get("OPENROUTER_API_KEY"):
        raise ValueError(f"OPENROUTER_API_KEY is missing from {path}")
    supplied_lock = values.get("CHRYS_MODEL_LOCK")
    if supplied_lock is not None and normalized_model_lock(supplied_lock) != expected_model_lock():
        raise ValueError("CHRYS_MODEL_LOCK does not match the pinned DeepSeek V4 Pro OpenRouter profile")
    values["CHRYS_MODEL_LOCK"] = expected_model_lock()
    return values


def write_json(path: Path, value: object) -> None:
    """Write stable JSON suitable for later provenance checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fingerprints_as_dict(items: list[TaskFingerprint]) -> list[dict[str, object]]:
    """Serialize task fingerprints without coupling callers to dataclasses."""
    return [asdict(item) for item in items]
