# Copyright (c) 2026 Chrys. All rights reserved.

"""Parallel requirement clarification and semantic-localization preflight."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from chrys.foundation.events.types import RequirementEnrichmentPhaseChanged, UserInjectResult, Warning
from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.kernel import Content, Message
from chrys.service.requirement_clarification.artifacts import ClarificationArtifactStore
from chrys.service.requirement_clarification.model import ChrysClarificationModel
from chrys.service.requirement_clarification.prompts import LEGACY_V1_STRATEGY_VERSION, STRATEGY_VERSION
from chrys.service.requirement_clarification.service import ClarificationService
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot, WorkspaceSnapshotter
from chrys.service.requirement_clarification.types import (
    ClarificationResult,
    ClarificationSelection,
    ClarificationStrategy,
    RequirementRevision,
)
from chrys.service.semantic_search import (
    LocalizationResult,
    SemanticSearchConfig,
    SemanticSearchError,
    SemanticSearchMode,
    localize_requirement_async,
)

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.models.workspace import Workspace
    from chrys.orchestration.engine.executor import Executor
    from chrys.orchestration.engine.run.runner import TurnRunner
    from chrys.orchestration.engine.run.turn_state import CurrentRunInjectionWindow, CurrentRunScope, TurnRuntimeState
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
    from chrys.service.profiles.models.schema import ModelProfile
    from chrys.service.session.history import SessionHistoryManager
    from chrys.service.trajectory.preparation import PreparationTrace


logger = logging.getLogger(__name__)

REQUIREMENT_ENRICHMENT_ARTIFACT_DIR = "requirement_enrichment"
_LOCALIZATION_CONTEXT_MAX_CHARS = 12_000


class RequirementEnrichmentHost(Protocol):
    """Engine surface used by a parallel enrichment workflow."""

    _active_profile: ModelProfile | None
    _bus: EventBus
    _executor: Executor
    _history: SessionHistoryManager
    _reminder_middleware: SystemReminderMiddleware | None
    _session_id: str | None
    _turn_number: int
    _turn_state: TurnRuntimeState
    _workspace: Workspace | None
    _requirement_enrichment_workflow: RequirementEnrichmentWorkflow | None

    @property
    def _session_dir(self) -> Path | None: ...

    def _accumulate_side_call_usage(self, usage_details: Mapping[str, Any]) -> None: ...

    async def persist_recovery_now(self) -> bool: ...


class RequirementEnrichmentWorkflow:
    """Derive two independent, frozen-workspace hints before one normal run."""

    def __init__(
        self,
        host: RequirementEnrichmentHost,
        runner: TurnRunner,
        *,
        strategy: ClarificationStrategy,
        clarification_timeout_seconds: float,
        localization_mode: SemanticSearchMode,
        localization_timeout_seconds: float,
        localization_model_profile: str,
    ) -> None:
        self._host = host
        self._runner = runner
        self._workflow_id = uuid4().hex
        self._snapshotter = WorkspaceSnapshotter()
        self._revision = RequirementRevision(number=1, messages=())
        self._phase_name = "snapshot"
        self._stop_requested = False
        self._analysis_tasks: tuple[asyncio.Task[Any], ...] = ()
        self._strategy = strategy
        self._clarification_timeout_seconds = clarification_timeout_seconds
        self._localization_mode = localization_mode
        self._localization_timeout_seconds = localization_timeout_seconds
        self._localization_model_profile = localization_model_profile
        self._initial_contents: list[Any] = []
        self._created_at: datetime | str | None = None

    @property
    def accepts_amendments(self) -> bool:
        """Return whether analysis can restart against amended user authority."""
        return self._phase_name in {"snapshot", "analyzing"}

    async def accept_amendment(
        self,
        text: str,
        *,
        created_at: datetime | str | None,
        injection_id: str | None,
    ) -> bool:
        """Append an amendment and cancel the stale pair of analyses."""
        if not self.accepts_amendments or not text:
            return False
        self._revision = self._revision.append(text)
        await self._checkpoint_revision()
        for task in self._analysis_tasks:
            task.cancel()
        await self._host._bus.publish(
            UserInjectResult(
                text=text,
                consumed=True,
                created_at=created_at,
                injection_id=injection_id,
                session_id=self._host._session_id,
            )
        )
        return True

    async def request_stop(self) -> None:
        """Cancel preflight work; the normal runner records the pending interrupt."""
        self._stop_requested = True
        for task in self._analysis_tasks:
            task.cancel()
        if self._host._executor.is_running:
            await self._host._executor.interrupt()

    async def run(
        self,
        text: str,
        *,
        created_at: datetime | str | None,
        contents: list[Any],
        run_scope: CurrentRunScope | None,
        injection_window: CurrentRunInjectionWindow | None,
        admission_preparation: PreparationTrace | None,
    ) -> None:
        host = self._host
        workspace = host._workspace
        model_profile = host._active_profile
        session_dir = host._session_dir
        if workspace is None or model_profile is None or session_dir is None or host._reminder_middleware is None:
            await self._warning("requirement enrichment runtime is unavailable")
            await self._run_standard(
                text,
                created_at=created_at,
                contents=contents,
                run_scope=run_scope,
                injection_window=injection_window,
                admission_preparation=admission_preparation,
            )
            return

        self._revision = RequirementRevision(number=1, messages=(text,))
        self._initial_contents = list(contents)
        self._created_at = created_at
        host._requirement_enrichment_workflow = self
        snapshot: WorkspaceSnapshot | None = None
        try:
            await self._checkpoint_revision()
            await self._phase("snapshot", 1)
            try:
                artifacts = await asyncio.to_thread(
                    ClarificationArtifactStore,
                    session_dir,
                    host._turn_number + 1,
                    artifact_dir_name=REQUIREMENT_ENRICHMENT_ARTIFACT_DIR,
                    artifact_subdir="clarification",
                )
            except Exception as exc:
                logger.warning("Requirement enrichment artifact initialization failed", exc_info=True)
                await self._warning(
                    f"requirement enrichment artifact initialization failed: {type(exc).__name__}: {exc}"[:1000]
                )
                self._phase_name = "executing"
                await self._run_standard(
                    self._revision.rendered,
                    created_at=created_at,
                    contents=_revision_contents(contents, self._revision),
                    run_scope=run_scope,
                    injection_window=injection_window,
                    admission_preparation=admission_preparation,
                )
                await self._phase("degraded", self._revision.number, terminal=True)
                return
            turn_artifact_root = artifacts.root.parent
            await self._save_artifact(
                "requirement input",
                lambda: artifacts.save_requirement_input(revision=1, messages=self._revision.messages),
            )
            try:
                snapshot = await asyncio.to_thread(
                    self._snapshotter.capture,
                    workspace,
                    artifacts.root / "s0",
                    snapshot_id=f"{self._workflow_id}-s0",
                    include_git_history=True,
                )
            except Exception as exc:
                logger.warning("Requirement enrichment snapshot failed", exc_info=True)
                await self._warning(f"requirement enrichment snapshot failed: {type(exc).__name__}: {exc}"[:1000])
                self._phase_name = "executing"
                await self._run_standard(
                    self._revision.rendered,
                    created_at=created_at,
                    contents=_revision_contents(contents, self._revision),
                    run_scope=run_scope,
                    injection_window=injection_window,
                    admission_preparation=admission_preparation,
                )
                await self._phase("degraded", self._revision.number, terminal=True)
                return

            while True:
                revision = self._revision
                history_background = _history_background(host._executor.history_state)
                await self._save_artifact(
                    "requirement input",
                    lambda revision=revision: artifacts.save_requirement_input(
                        revision=revision.number,
                        messages=revision.messages,
                    ),
                )
                await self._phase("analyzing", revision.number)
                clarification_task = asyncio.create_task(
                    self._clarify(
                        revision=revision,
                        background=history_background,
                        snapshot=snapshot,
                        profile=model_profile,
                        artifacts=artifacts,
                    )
                )
                localization_task = asyncio.create_task(
                    self._localize(revision=revision, snapshot=snapshot, artifacts=artifacts)
                )
                self._analysis_tasks = (clarification_task, localization_task)
                results = await asyncio.gather(*self._analysis_tasks, return_exceptions=True)
                self._analysis_tasks = ()
                # Amendment admission must close before any warning publication
                # can yield; otherwise a just-accepted revision can be consumed
                # after the restart check and never reach either analysis.
                self._phase_name = "assembling"
                if self._stop_requested:
                    await self._phase("interrupted", self._revision.number, terminal=True)
                    await self._run_standard(
                        self._revision.rendered,
                        created_at=created_at,
                        contents=_revision_contents(contents, self._revision),
                        run_scope=run_scope,
                        injection_window=injection_window,
                        admission_preparation=admission_preparation,
                    )
                    return
                if revision.number != self._revision.number:
                    continue
                clarification = _result_or_none(results[0], ClarificationResult)
                localization = _result_or_none(results[1], LocalizationResult)
                if isinstance(results[0], BaseException) and not isinstance(results[0], asyncio.CancelledError):
                    await self._warning(f"requirement clarification failed: {results[0]}"[:1000])
                elif clarification is not None and clarification.status != "completed":
                    detail = clarification.warnings[0] if clarification.warnings else clarification.empty_reason
                    await self._warning(f"requirement clarification degraded: {detail}"[:1000])
                if isinstance(results[1], BaseException) and not isinstance(results[1], asyncio.CancelledError):
                    await self._warning(f"requirement localization failed: {results[1]}"[:1000])
                break

            bundle = _render_bundle(clarification, localization)
            analysis_degraded = clarification is None or clarification.status != "completed" or localization is None
            await self._save_artifact(
                "enrichment bundle",
                lambda: _save_bundle(
                    turn_artifact_root / "bundle.json",
                    revision=revision,
                    bundle=bundle,
                    clarification=clarification,
                    localization=localization,
                ),
            )
            stale_guidance = False

            async def _queue_current_guidance() -> None:
                nonlocal stale_guidance
                if not bundle:
                    return
                if not await asyncio.to_thread(self._snapshotter.matches, snapshot):
                    stale_guidance = True
                    await self._warning("workspace changed during requirement enrichment; discarded stale guidance")
                    return
                host._reminder_middleware.queue_hook_reminders([bundle], for_next_turn=True)

            await self._phase("executing", revision.number)
            await self._run_standard(
                revision.rendered,
                created_at=created_at,
                contents=_revision_contents(contents, revision),
                run_scope=run_scope,
                injection_window=injection_window,
                admission_preparation=admission_preparation,
                before_execution=_queue_current_guidance,
            )
            terminal_phase = "degraded" if analysis_degraded or stale_guidance else "completed"
            await self._phase(terminal_phase, revision.number, terminal=True)
        finally:
            self._analysis_tasks = ()
            if host._requirement_enrichment_workflow is self:
                host._requirement_enrichment_workflow = None
            if snapshot is not None:
                await asyncio.to_thread(self._snapshotter.discard, snapshot)

    async def _clarify(
        self,
        *,
        revision: RequirementRevision,
        background: str,
        snapshot: WorkspaceSnapshot,
        profile: ModelProfile,
        artifacts: ClarificationArtifactStore,
    ) -> ClarificationResult:
        model = ChrysClarificationModel(
            profile=profile,
            snapshot=snapshot,
            session_id=self._host._session_id,
            session_dir=self._host._session_dir,
            report_usage=self._host._accumulate_side_call_usage,
        )
        service = ClarificationService(model, strategy=self._strategy)
        try:
            async with asyncio.timeout(self._clarification_timeout_seconds):
                result = await service.clarify(revision=revision, background=background, snapshot=snapshot)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"[:1000]
            result = ClarificationResult(
                strategy_version=LEGACY_V1_STRATEGY_VERSION
                if self._strategy == "legacy-v1-exact"
                else STRATEGY_VERSION,
                revision=revision.number,
                delta="",
                selection=ClarificationSelection(),
                status="degraded",
                empty_reason="clarification_failed",
                warnings=(f"clarification failed: {detail}",),
            )
        await self._save_artifact(
            "clarification result",
            lambda: artifacts.save_result(result, requirement_messages=revision.messages),
        )
        if result.status == "completed":
            try:
                async with asyncio.timeout(self._clarification_timeout_seconds):
                    pact_input, pact_usage = await service.generate_pact_input(
                        result=result,
                        revision=revision,
                        background=background,
                        snapshot=snapshot,
                    )
                result = replace(result, pact_input=pact_input, usage_details=(*result.usage_details, *pact_usage))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"[:1000]
                result = replace(
                    result,
                    pact_generation_error=detail,
                    warnings=(*result.warnings, f"PACT input generation failed: {detail}"),
                )
        await self._save_artifact(
            "clarification result",
            lambda: artifacts.save_result(result, requirement_messages=revision.messages),
        )
        await self._save_artifact("PACT generation", lambda: artifacts.save_pact_generation(result))
        return result

    async def _localize(
        self,
        *,
        revision: RequirementRevision,
        snapshot: WorkspaceSnapshot,
        artifacts: ClarificationArtifactStore,
    ) -> LocalizationResult:
        if not snapshot.roots:
            raise SemanticSearchError("requirement-enrichment snapshot has no workspace roots")
        ordered_roots = sorted(snapshot.roots, key=lambda root: not root.is_primary)
        artifact_root = artifacts.root.parent / "localization"
        async with asyncio.timeout(self._localization_timeout_seconds):
            tasks = [
                asyncio.create_task(
                    localize_requirement_async(
                        root.view_root,
                        revision.rendered,
                        artifact_dir=(artifact_root if len(ordered_roots) == 1 else artifact_root / f"root-{index}"),
                        config=SemanticSearchConfig(
                            mode=self._localization_mode,
                            timeout_seconds=self._localization_timeout_seconds,
                            model_profile=self._localization_model_profile,
                        ),
                    )
                )
                for index, root in enumerate(ordered_roots, start=1)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = [
            (root, result)
            for root, result in zip(ordered_roots, results, strict=True)
            if isinstance(result, LocalizationResult)
        ]
        if not successful:
            details = "; ".join(str(result) for result in results if isinstance(result, BaseException))
            raise SemanticSearchError(f"localization failed for every workspace root: {details}"[:1000])
        locations: list[dict[str, Any]] = []
        related_tests: list[str] = []
        related_files: list[str] = []
        unresolved_questions: list[str] = []
        warnings: list[str] = []
        for root, result in successful:
            root_label = root.label or ("primary" if root.is_primary else Path(root.source_root).name)
            locations.extend({**item, "workspace_root": root_label} for item in result.locations)
            for key, destination in (
                ("related_tests", related_tests),
                ("related_files", related_files),
                ("unresolved_questions", unresolved_questions),
            ):
                values = result.payload.get(key, [])
                if isinstance(values, list):
                    destination.extend(str(value) for value in values)
            warnings.extend(result.warnings)
        failed_roots = [
            root.label or Path(root.source_root).name
            for root, result in zip(ordered_roots, results, strict=True)
            if not isinstance(result, LocalizationResult)
        ]
        warnings.extend(f"localization failed for workspace root: {label}" for label in failed_roots)
        return LocalizationResult(
            payload={
                "locations": locations,
                "related_tests": list(dict.fromkeys(related_tests)),
                "related_files": list(dict.fromkeys(related_files)),
                "unresolved_questions": list(dict.fromkeys(unresolved_questions)),
            },
            artifacts=successful[0][1].artifacts,
            reused=all(result.reused for _, result in successful),
            warnings=warnings,
        )

    async def _run_standard(self, text: str, **kwargs: Any) -> None:
        await self._runner._run_fresh_standard(text, **kwargs)

    async def _warning(self, message: str) -> None:
        await self._host._bus.publish(
            Warning(code="requirement_enrichment_degraded", message=message, session_id=self._host._session_id)
        )

    async def _save_artifact(self, description: str, operation: Callable[[], None]) -> bool:
        try:
            await asyncio.to_thread(operation)
        except Exception as exc:
            logger.warning("Requirement enrichment %s artifact write failed", description, exc_info=True)
            await self._warning(
                f"requirement enrichment {description} artifact write failed: {type(exc).__name__}: {exc}"[:1000]
            )
            return False
        return True

    async def _checkpoint_revision(self) -> None:
        host = self._host
        revision = self._revision
        host._turn_state.history_start_index = len(host._history.messages)
        host._turn_state.set_current_input(
            revision.rendered,
            _revision_contents(self._initial_contents, revision),
            self._created_at,
        )
        try:
            await host.persist_recovery_now()
        except Exception as exc:
            logger.warning("Requirement enrichment input recovery checkpoint failed", exc_info=True)
            await self._warning(
                f"requirement enrichment input recovery checkpoint failed: {type(exc).__name__}: {exc}"[:1000]
            )

    async def _phase(self, phase: str, revision: int, *, terminal: bool = False) -> None:
        self._phase_name = phase
        await self._host._bus.publish(
            RequirementEnrichmentPhaseChanged(
                workflow_id=self._workflow_id,
                phase=phase,
                revision=revision,
                terminal=terminal,
                session_id=self._host._session_id,
            )
        )


def _result_or_none[T](value: object, expected: type[T]) -> T | None:
    return value if isinstance(value, expected) else None


def _history_background(history_state: dict[str, Any]) -> str:
    messages: list[Message] = []
    for block in history_state.get("compressed_msgs", []):
        block_messages = block.messages
        if isinstance(block_messages, list):
            messages.extend(message for message in block_messages if isinstance(message, Message))
    live = history_state.get("messages", [])
    if isinstance(live, list):
        messages.extend(message for message in live if isinstance(message, Message))
    rows = [
        f"{message.role}: {message.text}"
        for message in messages[-40:]
        if message.role in {"user", "assistant"}
        and message.text
        and not message.additional_properties.get("_chrys_kind")
    ]
    return "\n".join(rows)[-12_000:]


def _revision_contents(contents: list[Any], revision: RequirementRevision) -> list[Any]:
    if len(revision.messages) == 1:
        return contents
    amendments = "\n\nUser amendments, in authority order:\n" + "\n".join(
        f"{index}. {message}" for index, message in enumerate(revision.messages[1:], start=1)
    )
    return [*contents, Content.from_text(amendments)]


def _render_bundle(
    clarification: ClarificationResult | None,
    localization: LocalizationResult | None,
) -> str:
    sections = [
        "[REQUIREMENT_ENRICHMENT]",
        (
            "The original user requirement is authoritative. The following repository-derived material is advisory. "
            "Verify every location and claim against source before editing."
        ),
    ]
    if clarification is not None and clarification.status == "completed" and clarification.delta.strip():
        sections.extend(("", "## Requirement clarification", clarification.delta.strip()))
    localization_text = _render_localization(localization)
    if localization_text:
        sections.extend(("", "## Requirement localization", localization_text))
    return "\n".join(sections) if len(sections) > 2 else ""


def _render_localization(result: LocalizationResult | None) -> str:
    if result is None:
        return ""
    lines: list[str] = []
    for item in result.locations[:12]:
        path = str(item.get("file_path") or item.get("file") or "").strip()
        if not path:
            continue
        role = str(item.get("role", "candidate")).strip()
        symbol = str(item.get("symbol") or item.get("function_name") or item.get("class_name") or "").strip()
        reason = " ".join(str(item.get("reason", "")).split())[:600]
        start = item.get("start_line")
        end = item.get("end_line")
        workspace_root = str(item.get("workspace_root") or "").strip()
        location = f"{workspace_root}:{path}" if workspace_root else path
        if isinstance(start, int):
            location += f":{start}"
            if isinstance(end, int) and end != start:
                location += f"-{end}"
        if symbol:
            location += f" ({symbol})"
        line = f"- [{role}] {location}"
        if reason:
            line += f" — {reason}"
        lines.append(line)
    related_tests = result.payload.get("related_tests", [])
    if isinstance(related_tests, list):
        tests = [str(path).strip() for path in related_tests[:12] if str(path).strip()]
        if tests:
            lines.extend(("", "Related tests:", *(f"- {path}" for path in tests)))
    related_files = result.payload.get("related_files", [])
    if isinstance(related_files, list):
        files = [str(path).strip() for path in related_files[:12] if str(path).strip()]
        if files:
            lines.extend(
                ("", "Related configuration, build, and documentation files:", *(f"- {path}" for path in files))
            )
    unresolved = result.payload.get("unresolved_questions", [])
    if isinstance(unresolved, list):
        questions = [" ".join(str(question).split())[:600] for question in unresolved[:8] if str(question).strip()]
        if questions:
            lines.extend(("", "Unresolved questions:", *(f"- {question}" for question in questions)))
    return "\n".join(lines)[:_LOCALIZATION_CONTEXT_MAX_CHARS]


def _save_bundle(
    path: Path,
    *,
    revision: RequirementRevision,
    bundle: str,
    clarification: ClarificationResult | None,
    localization: LocalizationResult | None,
) -> None:
    payload = {
        "schema": "chrys/requirement-enrichment/v1",
        "revision": revision.number,
        "clarification_status": clarification.status if clarification is not None else "failed",
        "localization_status": "completed" if localization is not None else "failed",
        "bundle": bundle,
    }
    atomic_write_owner_only_text(path, json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
