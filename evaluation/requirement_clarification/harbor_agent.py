# Copyright (c) 2026 Chrys. All rights reserved.

"""Harbor adapter that runs a prebuilt Chrys binary inside DeepSWE containers."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from evaluation.requirement_clarification.protocol import (
    ADAPTER_MODES,
    MODEL_PROFILE_ID,
    expected_model_lock,
    normalized_model_lock,
    sha256_file,
    write_json,
)

_REMOTE_ROOT = "/tmp/chrys-evaluation"
_REMOTE_HOME = f"{_REMOTE_ROOT}/home"
_REMOTE_BINARY = f"{_REMOTE_ROOT}/bin/chrys"
_REMOTE_AGENT_PROFILE = f"{_REMOTE_HOME}/.chrys/agents/DeepSWEEvaluation.yaml"
_REMOTE_MODEL_PROFILE = f"{_REMOTE_HOME}/.chrys/models/{MODEL_PROFILE_ID}.yaml"


class ChrysHarborAgent(BaseAgent):
    """Upload and run a pinned offline Chrys build through Harbor."""

    def __init__(
        self,
        *args: Any,
        chrys_binary: str,
        agent_profile: str,
        model_profile: str,
        run_mode: str,
        chrys_revision: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if run_mode not in ADAPTER_MODES:
            raise ValueError(f"unsupported run_mode {run_mode!r}; expected one of {ADAPTER_MODES!r}")
        self._chrys_binary = Path(chrys_binary).resolve(strict=True)
        self._agent_profile = Path(agent_profile).resolve(strict=True)
        self._model_profile = Path(model_profile).resolve(strict=True)
        self._run_mode = run_mode
        self._chrys_revision = chrys_revision.strip()
        if not self._chrys_revision:
            raise ValueError("chrys_revision is required")

    @staticmethod
    @override
    def name() -> str:
        return "chrys-requirement-clarification"

    @override
    def version(self) -> str:
        return self._chrys_revision

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        setup = await environment.exec(
            "mkdir -p "
            f"{shlex.quote(_REMOTE_ROOT + '/bin')} "
            f"{shlex.quote(_REMOTE_HOME + '/.chrys/agents')} "
            f"{shlex.quote(_REMOTE_HOME + '/.chrys/models')} "
            f"{shlex.quote(EnvironmentPaths.agent_dir.as_posix())}",
            timeout_sec=30,
            user="root",
        )
        if setup.return_code != 0:
            raise RuntimeError(f"failed to create Chrys runtime directories: {setup.stderr or setup.stdout}")
        await environment.upload_file(self._chrys_binary, _REMOTE_BINARY)
        await environment.upload_file(self._agent_profile, _REMOTE_AGENT_PROFILE)
        await environment.upload_file(self._model_profile, _REMOTE_MODEL_PROFILE)
        permissions = await environment.exec(
            f"chmod 755 {shlex.quote(_REMOTE_BINARY)} && "
            f"chmod 644 {shlex.quote(_REMOTE_AGENT_PROFILE)} {shlex.quote(_REMOTE_MODEL_PROFILE)}",
            timeout_sec=30,
            user="root",
        )
        if permissions.return_code != 0:
            raise RuntimeError(f"failed to set Chrys runtime permissions: {permissions.stderr or permissions.stdout}")

    def _runtime_env(self) -> dict[str, str]:
        api_key = self._get_env("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        supplied_lock = self._get_env("CHRYS_MODEL_LOCK")
        if supplied_lock is not None and normalized_model_lock(supplied_lock) != expected_model_lock():
            raise ValueError("CHRYS_MODEL_LOCK does not match the evaluation model")
        return {
            "HOME": _REMOTE_HOME,
            "OPENROUTER_API_KEY": api_key,
            "CHRYS_MODEL_LOCK": expected_model_lock(),
            "CHRYS_SESSION_ROOT_DIR": f"{EnvironmentPaths.agent_dir.as_posix()}/chrys-sessions",
            "CHRYS_DEFAULT_APPROVAL_MODE": "bypass",
        }

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        instruction_path = self.logs_dir / "instruction.md"
        instruction_path.write_text(instruction, encoding="utf-8")
        remote_instruction = f"{EnvironmentPaths.agent_dir.as_posix()}/instruction.md"

        command = " ".join(
            shlex.quote(part)
            for part in (
                _REMOTE_BINARY,
                "run",
                "--task",
                remote_instruction,
                "--agent",
                "DeepSWEClarification" if self._run_mode == "clarification" else "DeepSWEControl",
                "--model",
                MODEL_PROFILE_ID,
                "--workdir",
                "/app",
                "--json",
            )
        )
        result = await environment.exec(command, env=self._runtime_env())
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        (self.logs_dir / "chrys.stdout.json").write_text(stdout, encoding="utf-8")
        (self.logs_dir / "chrys.stderr.log").write_text(stderr, encoding="utf-8")

        session_id: str | None = None
        if stdout.strip():
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("session_id"), str):
                session_id = payload["session_id"]

        metadata = {
            "protocol": "chrys-deepswe-requirement-clarification-v1",
            "run_mode": self._run_mode,
            "chrys_revision": self._chrys_revision,
            "chrys_binary_sha256": sha256_file(self._chrys_binary),
            "agent_profile_sha256": sha256_file(self._agent_profile),
            "model_profile_sha256": sha256_file(self._model_profile),
            "model_profile_id": MODEL_PROFILE_ID,
            "session_id": session_id,
            "return_code": result.return_code,
        }
        context.metadata = metadata
        write_json(self.logs_dir / "experiment.json", metadata)
        if result.return_code != 0:
            detail = stderr.strip()[-4000:] or stdout.strip()[-4000:] or "no output"
            raise RuntimeError(f"Chrys exited with code {result.return_code}: {detail}")
