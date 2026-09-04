# Copyright (c) 2026 Chrys. All rights reserved.

"""Run PACT roles through fresh in-process Chrys sessions."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import re
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Literal, Protocol

from acp import schema as acp_schema
from acp.helpers import start_tool_call, update_tool_call
from pact_core.review_decision import parse_review_decision
from pact_core.schemas import (
    AdapterProbe,
    ReviewDecisionCapture,
    TurnRequest,
    TurnResult,
    TurnStatus,
)

from chrys import __version__
from chrys.app.acp.bridge import AcpEventBridge, SessionUpdate
from chrys.foundation.config.context import EvalContext
from chrys.foundation.config.process_settings import reattribute_command_line, route_restart_settings
from chrys.foundation.config.settings_store import LoadedSettings, load_settings
from chrys.foundation.events.types import Event, TodoListUpdated
from chrys.orchestration.session_host import (
    Cancelled,
    ChrysSessionHost,
    EndTurn,
    Errored,
    TurnOutcome,
)
from chrys.service.approval.policy import ApprovalMode
from chrys.service.llm.json_extract import json_object_candidates, repair_json_object_candidate

SemanticRole = Literal["worker", "reviewer", "planner", "manager"]
SendUpdate = Callable[[SessionUpdate], Awaitable[None]]

_OUTPUT_SOURCE = "chrys_in_process"
# Bounds a role host's cancel and shutdown. A host still tears down MCP
# servers and drains hooks here; five seconds was not enough for a live one.
_DEFAULT_CLEANUP_GRACE_SECONDS = 30.0
_REVIEW_DECISION_RELATIVE_PATH = Path(".pact-io") / "reviewer-decision.json"
_REVIEW_TRANSPORT_EPILOGUE = """

---
Adapter transport requirement: write the machine-readable review decision to
`.pact-io/reviewer-decision.json` as exactly one JSON object. It must contain
`verdict` with value `complete`, `continue`, or `failed`. If the current route
must be reconsidered, use `continue` and include exactly one `plan_challenge`
object with non-empty `reason` and `gap_signature` strings and
`recommended_action` set to `replan`, `retry`, `request_user`, or `block`.
Do not add Markdown fences, prose, or any other fields to that JSON file. Your
ordinary final response remains the human-readable review evidence.
"""


class RoleTurnCancelled(RuntimeError):
    """Raised across the PACT adapter boundary when the outer ACP run aborts."""


class RoleUpdateError(RuntimeError):
    """Raised when role presentation can no longer reach the outer ACP client."""


class RoleRuntimeUnresponsive(RuntimeError):
    """Raised when an in-process role cannot be stopped within its cleanup bound."""


class _RoleHost(Protocol):
    @property
    def session_id(self) -> str | None: ...

    @property
    def last_turn_outcome(self) -> TurnOutcome | None: ...

    def iter_turn_events(self, message: str) -> AsyncIterator[Event]: ...

    async def cancel_current_turn(self) -> None: ...

    async def shutdown(self) -> None: ...


class _HostFactory(Protocol):
    def __call__(
        self,
        *,
        profile_name: str,
        loaded_settings: LoadedSettings,
        approval_mode: ApprovalMode,
        cwd: str,
        allow_user_interaction: bool,
    ) -> _RoleHost: ...


def _default_host_factory(
    *,
    profile_name: str,
    loaded_settings: LoadedSettings,
    approval_mode: ApprovalMode,
    cwd: str,
    allow_user_interaction: bool,
) -> _RoleHost:
    return ChrysSessionHost(
        profile_name=profile_name,
        loaded_settings=loaded_settings,
        approval_mode=approval_mode,
        cwd=cwd,
        allow_user_interaction=allow_user_interaction,
    )


def _derive_turn_settings(workdir: Path, base: LoadedSettings) -> LoadedSettings:
    """Load one role session under its own project trust domain."""
    candidate = load_settings(
        project_root=workdir,
        eval_context=EvalContext(
            frontend_default_max_transient_retries=base.settings.frontend_default_max_transient_retries
        ),
    )
    loaded, _deferred = route_restart_settings(reattribute_command_line(candidate, base), base)
    # A role host must never route: it already runs inside a campaign, and a
    # long-horizon decision here would try to start another one. It has no
    # memory of its own either: the campaign's outcome is deposited by the
    # session that delegated it, and a Manager turn that autostarted the
    # graph's MCP server and then flushed a deposit at session end blew
    # through the cleanup grace period -- the first live campaign died there.
    return dataclasses.replace(
        loaded,
        settings=dataclasses.replace(
            loaded.settings,
            routing_mode="off",
            memory_mcp_enabled=False,
            memory_writeback_on_session_end=False,
        ),
    )


def _not_applicable_review_decision() -> ReviewDecisionCapture:
    return ReviewDecisionCapture(
        verdict_status="not_applicable",
        plan_challenge_status="not_applicable",
    )


def _bounded_diagnostic(error: BaseException) -> str:
    detail = str(error).strip() or type(error).__name__
    return detail[-4000:]


_FENCED_WHOLE = re.compile(r"\A\s*```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)\r?\n[ \t]*```\s*\Z", re.DOTALL)


_JSON_PROTOCOL_ROLES = frozenset({"manager", "planner"})

# What the runtime's own prompts leave implicit and live campaigns tripped over: a
# Planner that merged two missions by deleting one ("Planner proposal cannot delete
# existing Missions"), a Planner that decorated missions with a `successors` field the
# schema rejects, a Manager that answered in prose. Said once more, plainly.
_ROLE_PROTOCOL_REMINDERS = {
    "planner": (
        "Protocol constraints (the runtime rejects proposals that break them):\n"
        "- Never delete or rename an existing mission. To replace one, add a NEW mission whose "
        "`supersedes` lists the old id and add a matching `supersede_mission` operation; the "
        "old mission stays in the graph.\n"
        "- A mission has exactly these fields: id, objective, target_ac_ids, dependencies, "
        "supersedes, verification_intent. No successors, no status, no notes.\n"
        '- `operations` items are exactly `{"op":"add_mission","mission_id":...}` or '
        '`{"op":"supersede_mission","mission_id":<old>,"replacement_mission_ids":[...]}`.\n'
        "- Reply with the JSON object as the text of your message: no prose before or after it, "
        "no Markdown fence, and never written to a file instead of the reply."
    ),
    "manager": (
        "Protocol constraints: reply with the JSON decision object as the text of your message "
        "-- no prose before or after it, no Markdown fence, never written to a file instead of "
        "the reply. Use `expected_plan_revision` and `expected_work_state_revision` exactly as "
        "given in the input."
    ),
}

# A Planner asked for a format repair once wrote the repaired object to a file and
# replied with nothing; the runtime parsed "" and blocked the campaign. One follow-up
# in the same session recovers that turn at the cost of a single model call.
_JSON_FOLLOWUP_PROMPT = (
    "Your previous message contained no JSON object in its text (an object written to a file "
    "does not reach the runtime). Reply now with exactly one JSON object as the message text: "
    "no prose, no Markdown fence. If you prepared the object in a file, paste its contents."
)
_JSON_FOLLOWUP_TIMEOUT_SECONDS = 600.0


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def _protocol_payload(text: str) -> str:
    """The JSON object a Manager or Planner reply carries, wherever the model put it.

    The protocols want a bare object. Models return it inside a fence, after a
    paragraph of reasoning, or both -- a Manager that wrote "The campaign is
    genuinely blocked: …" and then the correct decision in a ```json block was
    rejected as "not valid JSON", and the campaign ended. A reply with no JSON
    object at all is returned as it is, so the runtime's own repair pass still
    sees what the model said.
    """
    stripped = _unfenced(text).strip()
    try:
        json.loads(stripped)
    except ValueError:
        pass
    else:
        return stripped
    for candidate in json_object_candidates(text):
        try:
            payload = json.loads(repair_json_object_candidate(candidate))
        except ValueError, RecursionError:
            continue
        if isinstance(payload, dict):
            return json.dumps(payload)
    return text


def _unfenced(text: str) -> str:
    """Return the body of a response that is exactly one Markdown code fence.

    The Manager and Planner protocols want a bare JSON object; models keep
    wrapping it in ```json anyway, and the first live campaign was blocked
    with ``manager_protocol_error`` on a decision that was otherwise correct.
    A response with anything outside the fence is left alone.
    """
    match = _FENCED_WHOLE.match(text)
    return match.group(1) if match else text


class InProcessChrysAdapter:
    """Synchronous PACT adapter backed by a fresh Chrys host for every turn."""

    def __init__(
        self,
        *,
        semantic_role: SemanticRole,
        profile_name: str,
        outer_loop: asyncio.AbstractEventLoop,
        campaign_id: str,
        send_update: SendUpdate,
        abort_event: threading.Event,
        loaded_settings: LoadedSettings,
        host_factory: _HostFactory | None = None,
        cleanup_grace_seconds: float = _DEFAULT_CLEANUP_GRACE_SECONDS,
    ) -> None:
        if not profile_name.strip():
            raise ValueError("profile_name is required")
        if not campaign_id.strip():
            raise ValueError("campaign_id is required")
        if not math.isfinite(cleanup_grace_seconds) or cleanup_grace_seconds <= 0:
            raise ValueError("cleanup_grace_seconds must be finite and greater than zero")
        self.id = f"chrys-in-process-{semantic_role}"
        self.semantic_role = semantic_role
        self._profile_name = profile_name.strip()
        self._outer_loop = outer_loop
        self._campaign_id = campaign_id.strip()
        self._send_update = send_update
        self._abort_event = abort_event
        self._loaded_settings = loaded_settings
        self._host_factory = host_factory or _default_host_factory
        self._cleanup_grace_seconds = cleanup_grace_seconds
        self._active_host: _RoleHost | None = None
        self._active_turn_task: asyncio.Task[None] | None = None
        self._retained_tasks: set[asyncio.Task[None]] = set()
        self._turn_counter = 0
        self._turn_counter_lock = threading.Lock()

    def probe(self) -> AdapterProbe:
        """Describe the configured in-process runtime without starting a model turn."""
        return AdapterProbe(
            adapter_id=self.id,
            available=True,
            version=__version__,
            binary_name="in_process",
            profile=self._profile_name,
            isolation_mode="fresh_in_process_session",
        )

    def run_turn(self, request: TurnRequest) -> TurnResult:
        """Schedule one Chrys role turn on the outer ACP event loop and block PACT."""
        sequence = self._next_turn_sequence()
        future = asyncio.run_coroutine_threadsafe(
            self._run_turn_async(request, sequence=sequence),
            self._outer_loop,
        )
        return future.result()

    async def cancel_current_turn(self) -> None:
        """Abort this Campaign invocation and interrupt its active role, if any."""
        self._abort_event.set()
        host = self._active_host
        if host is not None:
            cancel_task = asyncio.create_task(host.cancel_current_turn())
            if not await self._wait_for_task(cancel_task, timeout=self._cleanup_grace_seconds):
                raise self._runtime_unresponsive(
                    "Chrys role cancellation did not respond within the cleanup grace period",
                    cancel_task,
                    self._active_turn_task,
                )
            cancel_task.result()

    def _next_turn_sequence(self) -> int:
        with self._turn_counter_lock:
            self._turn_counter += 1
            return self._turn_counter

    async def _run_turn_async(self, request: TurnRequest, *, sequence: int) -> TurnResult:
        if self._abort_event.is_set():
            raise RoleTurnCancelled("PACT role turn cancelled before start")
        self._validate_request_role(request)
        namespace = f"pact:{self._campaign_id}:{self.semantic_role}:{sequence}"
        role_call_id = f"{namespace}:role"
        started = time.monotonic()
        await self._publish_update(
            start_tool_call(
                role_call_id,
                f"PACT {self.semantic_role.title()} turn",
                kind="think",
                status="in_progress",
                raw_input={"role": self.semantic_role},
            )
        )

        host: _RoleHost | None = None
        status: TurnStatus = "spawn_failed"
        final_text = ""
        diagnostic = ""
        session_id: str | None = None
        review_decision = _not_applicable_review_decision()
        transport_path = request.workdir / _REVIEW_DECISION_RELATIVE_PATH
        turn_task: asyncio.Task[None] | None = None
        preserve_runtime = False
        try:
            prompt = request.prompt
            if self.semantic_role in _ROLE_PROTOCOL_REMINDERS:
                prompt = f"{prompt}\n\n{_ROLE_PROTOCOL_REMINDERS[self.semantic_role]}"
            if self.semantic_role == "reviewer":
                self._clear_review_transport(transport_path)
                prompt += _REVIEW_TRANSPORT_EPILOGUE
            turn_settings = await asyncio.to_thread(
                _derive_turn_settings,
                request.workdir,
                self._loaded_settings,
            )
            if self._abort_event.is_set():
                raise RoleTurnCancelled("PACT role turn cancelled during setup")
            host = self._host_factory(
                profile_name=self._profile_name,
                loaded_settings=turn_settings,
                approval_mode=ApprovalMode.BYPASS,
                cwd=str(request.workdir),
                allow_user_interaction=False,
            )
            self._active_host = host
            if self._abort_event.is_set():
                await self.cancel_current_turn()
                raise RoleTurnCancelled("PACT role turn cancelled during setup")
            # Keep timeout ownership outside the consumer task. Cancelling that
            # task directly can block forever in ChrysSessionHost's shielded
            # async-generator cleanup while its engine run is still alive.
            turn_task = asyncio.create_task(self._consume_turn(host, prompt=prompt, namespace=namespace))
            self._active_turn_task = turn_task
            if not await self._wait_for_task(turn_task, timeout=request.timeout_seconds):
                try:
                    await self._cancel_and_drain_timed_out_turn(host, turn_task)
                except RoleRuntimeUnresponsive:
                    preserve_runtime = True
                    raise
                status = "timeout"
                diagnostic = "TimeoutError"
            else:
                try:
                    turn_task.result()
                except asyncio.CancelledError as error:
                    raise RoleTurnCancelled("PACT role turn was cancelled") from error
                status, final_text, diagnostic = self._map_outcome(host.last_turn_outcome)
                if status == "completed" and self.semantic_role in _JSON_PROTOCOL_ROLES:
                    final_text = _protocol_payload(final_text)
                if (
                    self.semantic_role in _JSON_PROTOCOL_ROLES
                    and status in ("completed", "output_missing")
                    and not _is_json_object(final_text)
                ):
                    turn_task = asyncio.create_task(
                        self._consume_turn(host, prompt=_JSON_FOLLOWUP_PROMPT, namespace=f"{namespace}:followup")
                    )
                    self._active_turn_task = turn_task
                    followup_timeout = min(request.timeout_seconds, _JSON_FOLLOWUP_TIMEOUT_SECONDS)
                    if not await self._wait_for_task(turn_task, timeout=followup_timeout):
                        try:
                            await self._cancel_and_drain_timed_out_turn(host, turn_task)
                        except RoleRuntimeUnresponsive:
                            preserve_runtime = True
                            raise
                        status = "timeout"
                        diagnostic = "TimeoutError"
                    else:
                        try:
                            turn_task.result()
                        except asyncio.CancelledError as error:
                            raise RoleTurnCancelled("PACT role turn was cancelled") from error
                        status, final_text, diagnostic = self._map_outcome(host.last_turn_outcome)
                        if status == "completed":
                            final_text = _protocol_payload(final_text)
            session_id = host.session_id
            if self.semantic_role == "reviewer":
                review_decision = self._capture_review_decision(request, transport_path)
        except RoleRuntimeUnresponsive:
            preserve_runtime = True
            raise
        except RoleUpdateError:
            raise
        except RoleTurnCancelled:
            await self._emit_role_terminal(role_call_id, status="cancelled")
            raise
        except Exception as error:
            status = "spawn_failed"
            diagnostic = _bounded_diagnostic(error)
            if self.semantic_role == "reviewer":
                review_decision = self._capture_review_decision_after_failure(request, transport_path, error)
        finally:
            if host is not None and not preserve_runtime:
                try:
                    await self._shutdown_host_bounded(host, turn_task=turn_task)
                except RoleRuntimeUnresponsive:
                    preserve_runtime = True
                    raise
                except Exception as error:
                    status = "spawn_failed"
                    diagnostic = _bounded_diagnostic(error)
                finally:
                    if not preserve_runtime:
                        if self._active_host is host:
                            self._active_host = None
                        if self._active_turn_task is turn_task:
                            self._active_turn_task = None

        review_decision = self._trust_review_decision(review_decision, turn_status=status)
        result = TurnResult(
            status=status,
            final_text=final_text if status == "completed" else "",
            output_source=_OUTPUT_SOURCE,
            exit_code=0 if status == "completed" else None,
            stderr_tail=diagnostic,
            duration_seconds=time.monotonic() - started,
            session_id=session_id,
            review_decision=review_decision,
        )
        await self._emit_role_terminal(role_call_id, status=result.status)
        return result

    async def _consume_turn(self, host: _RoleHost, *, prompt: str, namespace: str) -> None:
        bridge = AcpEventBridge()
        async for event in host.iter_turn_events(prompt):
            if self._abort_event.is_set():
                await self.cancel_current_turn()
            if isinstance(event, TodoListUpdated):
                continue
            try:
                updates = bridge.updates_for_event(event)
            except Exception as error:
                raise RoleUpdateError("Failed to project a Chrys event to ACP") from error
            for update in updates:
                try:
                    namespaced = self._namespace_tool_update(update, namespace)
                except Exception as error:
                    raise RoleUpdateError("Failed to namespace a Chrys ACP update") from error
                await self._publish_update(namespaced)
        if self._abort_event.is_set():
            raise RoleTurnCancelled("PACT role turn cancelled")

    async def _cancel_and_drain_timed_out_turn(
        self,
        host: _RoleHost,
        turn_task: asyncio.Task[None],
    ) -> None:
        deadline = asyncio.get_running_loop().time() + self._cleanup_grace_seconds
        cancel_task = asyncio.create_task(host.cancel_current_turn())
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        if not await self._wait_for_task(cancel_task, timeout=remaining):
            raise self._runtime_unresponsive(
                "Chrys role cancellation did not respond within the cleanup grace period",
                cancel_task,
                turn_task,
            )
        try:
            cancel_task.result()
        except Exception as error:
            raise self._runtime_unresponsive(
                f"Chrys role cancellation failed: {_bounded_diagnostic(error)}",
                turn_task,
            ) from error

        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        if not await self._wait_for_task(turn_task, timeout=remaining):
            raise self._runtime_unresponsive(
                "Chrys role turn did not stop within the cleanup grace period",
                turn_task,
            )
        self._observe_task(turn_task)

    async def _shutdown_host_bounded(
        self,
        host: _RoleHost,
        *,
        turn_task: asyncio.Task[None] | None,
    ) -> None:
        shutdown_task = asyncio.create_task(host.shutdown())
        if not await self._wait_for_task(shutdown_task, timeout=self._cleanup_grace_seconds):
            raise self._runtime_unresponsive(
                "Chrys role runtime did not shut down within the cleanup grace period",
                shutdown_task,
                turn_task,
            )
        shutdown_task.result()

    @staticmethod
    async def _wait_for_task(task: asyncio.Task[None], *, timeout: float) -> bool:
        if task.done():
            return True
        if timeout <= 0:
            return False
        done, _pending = await asyncio.wait({task}, timeout=timeout)
        return task in done

    def _runtime_unresponsive(
        self,
        message: str,
        *tasks: asyncio.Task[None] | None,
    ) -> RoleRuntimeUnresponsive:
        self._abort_event.set()
        for task in tasks:
            if task is not None and not task.done():
                self._retain_task(task)
        return RoleRuntimeUnresponsive(message)

    def _retain_task(self, task: asyncio.Task[None]) -> None:
        if task in self._retained_tasks:
            return
        self._retained_tasks.add(task)

        def settled(settled_task: asyncio.Task[None]) -> None:
            self._retained_tasks.discard(settled_task)
            self._observe_task(settled_task)
            if self._active_turn_task is settled_task:
                self._active_turn_task = None

        task.add_done_callback(settled)

    @staticmethod
    def _observe_task(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except BaseException:
            return

    def _validate_request_role(self, request: TurnRequest) -> None:
        expected = "reviewer" if self.semantic_role == "reviewer" else "worker"
        if request.role != expected:
            raise ValueError(f"{self.semantic_role} adapter expected a {expected} TurnRequest, got {request.role}")

    @staticmethod
    def _map_outcome(outcome: TurnOutcome | None) -> tuple[TurnStatus, str, str]:
        if isinstance(outcome, EndTurn):
            if outcome.final_text.strip():
                return "completed", _unfenced(outcome.final_text), ""
            return "output_missing", "", "Chrys role turn returned an empty final response"
        if isinstance(outcome, Cancelled):
            raise RoleTurnCancelled(outcome.reason or "PACT role turn cancelled")
        if isinstance(outcome, Errored):
            detail = outcome.error.message or outcome.error.code or "Chrys role turn failed"
            if outcome.error.code == "no_final_response":
                return "output_missing", "", detail
            return "spawn_failed", "", detail
        return "output_missing", "", "Chrys role turn ended without a terminal outcome"

    @staticmethod
    def _namespace_tool_update(update: SessionUpdate, namespace: str) -> SessionUpdate:
        if isinstance(update, acp_schema.ToolCallStart | acp_schema.ToolCallProgress):
            return update.model_copy(update={"tool_call_id": f"{namespace}:inner:{update.tool_call_id}"})
        return update

    async def _emit_role_terminal(self, role_call_id: str, *, status: str) -> None:
        completed = status == "completed"
        summary = f"PACT {self.semantic_role.title()} turn {status}."
        await self._publish_update(
            update_tool_call(
                role_call_id,
                status="completed" if completed else "failed",
                raw_output=summary,
            )
        )

    async def _publish_update(self, update: SessionUpdate) -> None:
        try:
            await self._send_update(update)
        except Exception as error:
            raise RoleUpdateError("Failed to send a Chrys role update over ACP") from error

    @staticmethod
    def _clear_review_transport(path: Path) -> None:
        """Remove a stale decision file, tolerating whatever is actually there.

        The agent owns this directory, so the path can be a directory or an
        unreadable entry rather than the file we expect. Failing to clear it is
        not a reason to refuse to start the turn -- the capture step reports a
        decision it cannot read.
        """
        try:
            path.unlink()
        except OSError:
            return

    @staticmethod
    def _capture_review_decision(request: TurnRequest, path: Path) -> ReviewDecisionCapture:
        existed = path.exists() or path.is_symlink()
        raw_text = ""
        if existed:
            if path.is_symlink():
                path.unlink()
            elif path.is_file():
                try:
                    raw_text = path.read_text(encoding="utf-8")
                except OSError, UnicodeError:
                    raw_text = ""
                finally:
                    path.unlink()
            else:
                # A directory (or anything else) where the decision file
                # belongs means the reviewer wrote no decision, which is what
                # `parse_review_decision` reports from empty text. Raising here
                # instead was caught as a spawn failure and threw away the
                # final text of a turn that had actually completed.
                raw_text = ""
        if existed and request.artifact_dir is not None:
            raw_path = request.artifact_dir / "reviewer-decision-raw.txt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(raw_text, encoding="utf-8")
        return parse_review_decision(raw_text, existed=existed)

    @classmethod
    def _capture_review_decision_after_failure(
        cls,
        request: TurnRequest,
        path: Path,
        error: BaseException,
    ) -> ReviewDecisionCapture:
        try:
            return cls._capture_review_decision(request, path)
        except Exception as capture_error:
            return ReviewDecisionCapture(
                verdict_status="malformed",
                plan_challenge_status="unavailable",
                error=(
                    "review decision capture failed after role failure: "
                    f"{_bounded_diagnostic(capture_error)}; role error: {_bounded_diagnostic(error)}"
                ),
            )

    @staticmethod
    def _trust_review_decision(
        capture: ReviewDecisionCapture,
        *,
        turn_status: TurnStatus,
    ) -> ReviewDecisionCapture:
        if turn_status == "completed" or capture.verdict_status != "valid":
            return capture
        return ReviewDecisionCapture(
            verdict_status="malformed",
            plan_challenge_status="unavailable",
            raw_text=capture.raw_text,
            error=f"decision is not credible because turn status was {turn_status}",
            plan_challenge_error=("plan challenge is not credible because the reviewer turn did not complete"),
        )
