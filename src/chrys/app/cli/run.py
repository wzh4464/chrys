# Copyright (c) 2026 Chrys. All rights reserved.

"""Headless ``chrys run`` command."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from chrys.app.features.buddy.lifecycle import on_successful_turn as on_buddy_successful_turn
from chrys.app.parsing import SanitizingArgumentParser
from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.config.context import EvalContext
from chrys.foundation.config.settings import (
    HEADLESS_DEFAULT_MAX_TRANSIENT_RETRIES,
    MAX_TRANSIENT_RETRIES_LIMIT,
    Settings,
)
from chrys.foundation.config.settings_store import LoadedSettings, SettingsWarning
from chrys.foundation.config.spec import ENV_SOURCES, Source
from chrys.foundation.config.warnings import settings_warning_events
from chrys.foundation.events.types import Warning
from chrys.foundation.i18n import DisplaySequence, Localizer, MessageRef, msg
from chrys.foundation.i18n.formatting import format_message, sanitize_legacy_scalar
from chrys.foundation.text.encoding import decode_bytes
from chrys.orchestration.session_host import (
    AgentProfileNotFoundError,
    AmbiguousSessionIdError,
    ChrysSessionHost,
    HeadlessRunError,
    HeadlessRunResult,
    SessionNotFoundError,
)
from chrys.orchestration.startup import bootstrap_runtime
from chrys.service.approval.policy import ApprovalMode
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.resolver import (
    format_available_profile_labels,
    loaded_with_active_model_profile,
    resolve_profile_selector,
)
from chrys.service.semantic_search import (
    SemanticSearchConfig,
    SemanticSearchError,
    SemanticSearchMode,
    localize_requirement,
)
from chrys.service.semantic_search.output import load_report

_MAX_TRANSIENT_RETRIES_INVALID = msg(
    "settings.max_transient_retries_invalid",
    fallback=(
        "Ignoring invalid CHRYS_MAX_TRANSIENT_RETRIES={raw}; "
        "expected a non-negative integer and will use the frontend default."
    ),
)
_MAX_TRANSIENT_RETRIES_CLAMPED = msg(
    "settings.max_transient_retries_clamped",
    fallback=("CHRYS_MAX_TRANSIENT_RETRIES={value} exceeds the limit of {limit}; clamping to {limit}."),
)
_HEADLESS_RUN_TIMEOUT = msg(
    "run.headless_timeout",
    fallback="Agent run timed out.",
)
_INTERRUPTED = msg(
    "run.interrupted",
    fallback="Interrupted by user.",
)
_MODEL_PROFILE_NOT_FOUND = msg(
    "run.model_profile_not_found",
    fallback="Model profile not found: {model}",
)
_MODEL_PROFILE_NOT_FOUND_WITH_AVAILABLE = msg(
    "run.model_profile_not_found_with_available",
    fallback="Model profile not found: {model}. Available model profiles: {available}",
)


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedRuntime:
    """Settings, localization, and warnings established by runtime preparation."""

    loaded: LoadedSettings
    localizer: Localizer
    pending_warnings: list[Warning]

    @property
    def settings(self) -> Settings:
        """The assembled settings, for callers that need nothing else."""
        return self.loaded.settings


@dataclasses.dataclass(slots=True)
class PreparedRuntimeHolder:
    """Per-invocation handoff from ``run_command`` to ``main`` handlers."""

    runtime: PreparedRuntime | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the ``chrys run`` argument parser."""
    parser = SanitizingArgumentParser(
        prog="chrys run",
        description=f"Run an {APP_DISPLAY_NAME} agent headlessly until the final response.",
        add_help=False,
    )
    parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit"
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to send to the agent")
    parser.add_argument(
        "-t",
        "--task",
        metavar="FILE",
        default=None,
        help="Read prompt from text file (encoding auto-detected, resolved relative to --workdir)",
    )
    parser.add_argument("-a", "--agent", required=True, help="Agent profile id, name, or display name to run")
    parser.add_argument(
        "-m",
        "--model",
        metavar="MODEL",
        default=None,
        help="Active model profile id or name to use as the fallback model for this run",
    )
    parser.add_argument("-s", "--session", default=None, help="Optional session id to restore before running")
    parser.add_argument(
        "-C",
        "--workdir",
        metavar="DIR",
        dest="cwd",
        default=None,
        help="Working directory for the run",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    parser.add_argument(
        "--semantic-localization",
        choices=[mode.value for mode in SemanticSearchMode],
        default="off",
        help="Run semantic code localization before the agent turn",
    )
    parser.add_argument(
        "--localization-output",
        metavar="FILE",
        help="Copy the generated Markdown localization report to this path",
    )
    parser.add_argument(
        "--localization-artifact-dir",
        metavar="DIR",
        help="Directory inside the workspace for localization artifacts",
    )
    parser.add_argument(
        "--localization-file",
        metavar="FILE",
        help="Use an existing Markdown localization report for this run",
    )
    parser.add_argument("--semantic-refresh", action="store_true", help="Ignore a matching localization cache")
    parser.add_argument("--localization-model-profile", default="", help="Model profile for LLM localization")
    parser.add_argument("--codegraph-command", default="", help="Optional CodeGraph command override")
    return parser


class TaskFileError(Exception):
    """Raised when a ``chrys run --task`` file cannot be loaded."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class ModelProfileNotFoundError(KeyError):
    """Raised when an explicit ``chrys run --model`` selector cannot be resolved."""

    def __init__(self, message: str, *, display_message: MessageRef | None = None) -> None:
        self.display_message = display_message
        super().__init__(message)


_RETRY_KEY = "llm.retry.max_transient"


def _speaks_for_the_env_var(warning: SettingsWarning) -> bool:
    """Whether this verdict gets the wording headless callers have been parsing.

    Only where the value really is spelled ``CHRYS_MAX_TRANSIENT_RETRIES``.
    The same key rejected in ``settings.yaml`` is a different thing for the
    user to go and fix, and telling them to check an environment variable they
    never set sends them looking in the wrong place. One predicate for both the
    skip and the report below, so the two cannot drift into double-reporting a
    warning or dropping one.
    """
    return warning.key == _RETRY_KEY and warning.origin.layer in ENV_SOURCES


def _retry_warning_ref(warning: SettingsWarning) -> MessageRef:
    if warning.rejected:
        return _MAX_TRANSIENT_RETRIES_INVALID.bind(raw=repr(warning.outcome.raw))
    # ``outcome.value`` is the post-clamp value (50); this message reports what
    # the user wrote ("999 exceeds the limit of 50"), so it binds the raw text.
    return _MAX_TRANSIENT_RETRIES_CLAMPED.bind(
        value=warning.outcome.raw.strip(),
        limit=MAX_TRANSIENT_RETRIES_LIMIT,
    )


def _prepare_runtime(*, restoring_session: bool = False) -> PreparedRuntime:
    """Load environment, apply runtime patches, and prepare display state without writing stderr.

    A ``--session`` restore loads its settings from the saved session's own
    root, which need not be the process cwd — bootstrapping project-free keeps
    the pending warnings from describing a repository the run doesn't live in.
    A failed restore aborts the run outright, so no fallback session ever runs
    on these project-free settings.
    """
    bootstrap = bootstrap_runtime(
        dotenv_override=True,
        configure_stdio=True,
        eval_context=EvalContext(
            frontend_default_max_transient_retries=HEADLESS_DEFAULT_MAX_TRANSIENT_RETRIES,
        ),
        project_root=None if restoring_session else Path(os.getcwd()),
    )
    localizer = Localizer("en")
    # Every warning composed here is root-independent — environment and user
    # layers, plus ``settle_session_root``'s verdict — so the list is right for
    # a restored session too, whatever root it lives in. The target root's own
    # additions (its project layer) are printed by ``run_command`` after the
    # restore, as the delta over this list.
    # This one key keeps its own wording, which predates the shared composer and
    # is what headless callers have been parsing; everything else gets the
    # generic message rather than being dropped.
    pending_warnings = [
        *bootstrap.warnings,
        *settings_warning_events(bootstrap.loaded, skip=_speaks_for_the_env_var),
    ]

    for warning in bootstrap.loaded.warnings:
        if not _speaks_for_the_env_var(warning):
            continue
        display_message = _retry_warning_ref(warning)
        pending_warnings.append(
            Warning(
                code="invalid_max_transient_retries",
                message=format_message(display_message),
                display_message=display_message,
            )
        )

    return PreparedRuntime(
        loaded=bootstrap.loaded,
        localizer=localizer,
        pending_warnings=pending_warnings,
    )


def _apply_active_model_selection(
    loaded: LoadedSettings,
    model_profile: str | None,
) -> tuple[LoadedSettings, ModelProfileRegistry | None]:
    model = (model_profile or "").strip()
    if not model:
        return loaded, None
    registry = ModelProfileRegistry()
    registry.load_all()
    profile = resolve_profile_selector(registry, model)
    if profile is None:
        error_message = f"Model profile not found: {model}"
        available = format_available_profile_labels(registry)
        if available:
            error_message = f"{error_message}. Available model profiles: {available}"
            profiles = sorted(registry.list_profiles(), key=lambda item: item.name.casefold())
            labels = DisplaySequence(f"{item.name} ({item.id})" for item in profiles)
            display_message = _MODEL_PROFILE_NOT_FOUND_WITH_AVAILABLE.bind(model=model, available=labels)
        else:
            display_message = _MODEL_PROFILE_NOT_FOUND.bind(model=model)
        raise ModelProfileNotFoundError(error_message, display_message=display_message)
    return loaded_with_active_model_profile(loaded, profile, Source.CLI), registry


def _configure_logging() -> None:
    """Prevent library logs from writing to stderr by default in headless CLI mode."""
    logging.basicConfig(handlers=[logging.NullHandler()])


def _write_result(result: HeadlessRunResult, *, as_json: bool, duration: float) -> None:
    if as_json:
        payload = {
            "session_id": result.session_id,
            "result": result.text,
            "duration": round(duration, 3),
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False))
        sys.stdout.write("\n")
        return
    sys.stdout.write(result.text)
    if not result.text.endswith("\n"):
        sys.stdout.write("\n")


def _write_error(message: str, *, as_json: bool, code: str = "error", session_id: str | None = None) -> None:
    if as_json:
        payload = {"error": message, "code": code}
        if session_id:
            # Failed runs still persist their session; surfacing the id lets a
            # batch runner locate and export the trajectory from JSON output
            # alone.
            payload["session_id"] = session_id
        sys.stderr.write(json.dumps(payload, ensure_ascii=False))
        sys.stderr.write("\n")
        return
    sys.stderr.write(f"Error: {sanitize_legacy_scalar(message)}\n")


def _write_warning(message: str, *, as_json: bool, code: str = "warning") -> None:
    if as_json:
        sys.stderr.write(json.dumps({"warning": message, "code": code}, ensure_ascii=False))
        sys.stderr.write("\n")
        return
    sys.stderr.write(f"Warning: {sanitize_legacy_scalar(message)}\n")


def _write_warning_events(warnings: Iterable[Warning], localizer: Localizer, *, as_json: bool) -> None:
    for warning in warnings:
        if as_json:
            message = warning.message
        elif warning.display_message is not None:
            message = localizer.render(warning.display_message)
        else:
            message = warning.message
        _write_warning(message, code=warning.code, as_json=as_json)


def _write_pending_warnings(runtime: PreparedRuntime, *, as_json: bool) -> None:
    _write_warning_events(runtime.pending_warnings, runtime.localizer, as_json=as_json)


def _apply_cwd(cwd: str | None) -> str | None:
    if not cwd:
        return None
    path = Path(cwd).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        message = f"Working directory does not exist: {cwd}"
        raise FileNotFoundError(message) from exc
    if not resolved.is_dir():
        message = f"Working directory is not a directory: {resolved}"
        raise NotADirectoryError(message)
    os.chdir(resolved)
    return os.fspath(resolved)


def _read_task_file(task: str) -> str:
    if task == "-":
        raise TaskFileError("Task file does not exist: -", code="task_file_not_found")

    path = Path(task).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        message = f"Task file does not exist: {task}"
        raise TaskFileError(message, code="task_file_not_found") from exc
    except OSError as exc:
        message = f"Failed to read task file: {path}: {exc}"
        raise TaskFileError(message, code="task_file_read_failed") from exc

    if not resolved.is_file():
        message = f"Task path is not a file: {resolved}"
        raise TaskFileError(message, code="task_file_not_file")

    try:
        raw = resolved.read_bytes()
    except FileNotFoundError as exc:
        message = f"Task file does not exist: {task}"
        raise TaskFileError(message, code="task_file_not_found") from exc
    except IsADirectoryError as exc:
        message = f"Task path is not a file: {resolved}"
        raise TaskFileError(message, code="task_file_not_file") from exc
    except OSError as exc:
        message = f"Failed to read task file: {resolved}: {exc}"
        raise TaskFileError(message, code="task_file_read_failed") from exc

    return decode_bytes(raw)


def _resolve_prompt(args: argparse.Namespace) -> str:
    if args.task is not None:
        return _read_task_file(args.task)
    if args.prompt is None:
        message = "Prompt source was not validated."
        raise ValueError(message)
    return args.prompt


def _append_localization_context(prompt: str, report_path: Path | None) -> str:
    """Append a bounded localization report to the user prompt, when present.

    Localization is a CLI preflight. Keeping its output in the prompt avoids
    adding localization-specific state to Chrys' engine and session layers.
    """
    if report_path is None:
        return prompt
    try:
        report = load_report(report_path)
    except OSError as exc:
        raise SemanticSearchError(f"failed to read localization report: {report_path}: {exc}") from exc
    if not report.strip():
        return prompt
    return (
        f"{prompt.rstrip()}\n\n"
        "<semantic-code-localization>\n"
        "The following report contains inspection candidates only. The original user requirement is authoritative. "
        "Read and verify source before editing; do not edit every listed file.\n\n"
        f"{report}\n"
        "</semantic-code-localization>"
    )


def _restore_delta_warnings(loaded: LoadedSettings, pending: Iterable[Warning]) -> list[Warning]:
    """Warnings the restored session's settings load adds over the bootstrap's.

    The two loads share every root-independent layer (environment, user
    document), so their verdicts overlap almost entirely; the delta is what
    the project-free bootstrap could not see — the target root's project
    layer and dormant project files. Comparing the composed events keeps the
    overlap out while leaving the already-printed pending list — with its
    settle verdict and its compatibility retry wording — untouched.
    """
    already = {(warning.code, warning.message) for warning in pending}
    return [
        warning
        for warning in settings_warning_events(loaded, skip=_speaks_for_the_env_var)
        if (warning.code, warning.message) not in already
    ]


async def run_command(args: argparse.Namespace, holder: PreparedRuntimeHolder) -> int:
    """Execute parsed ``chrys run`` args."""
    cwd = _apply_cwd(args.cwd)
    prompt = _resolve_prompt(args)
    localization_report: Path | None = None
    if args.localization_file:
        candidate = Path(args.localization_file).expanduser().resolve()
        if not candidate.is_file():
            raise SemanticSearchError(f"localization report does not exist: {candidate}")
        localization_report = candidate
    if args.semantic_localization != SemanticSearchMode.OFF.value:
        try:
            localization = localize_requirement(
                Path.cwd(),
                prompt,
                artifact_dir=args.localization_artifact_dir,
                config=SemanticSearchConfig(
                    mode=SemanticSearchMode(args.semantic_localization),
                    model_profile=args.localization_model_profile,
                ),
                refresh=args.semantic_refresh,
                codegraph_command=args.codegraph_command,
            )
            localization_report = localization.artifacts.report_markdown
            if args.localization_output:
                destination = Path(args.localization_output).expanduser().resolve()
                try:
                    destination.relative_to(Path.cwd().resolve())
                except ValueError as exc:
                    raise SemanticSearchError("--localization-output must be inside the workspace") from exc
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(localization_report.read_text(encoding="utf-8"), encoding="utf-8")
                localization_report = destination
        except SemanticSearchError as exc:
            if args.semantic_localization == SemanticSearchMode.LLM.value:
                raise
            _write_warning(
                f"Semantic localization unavailable: {exc}",
                as_json=args.json,
                code="semantic_localization",
            )
    prompt = _append_localization_context(prompt, localization_report)
    # Normalized once, to the host's own reading of the flag: the host strips
    # the id and treats a blank one as "no session", and a project-free
    # bootstrap for a run that then starts fresh would silently drop the
    # working directory's project layer.
    session_id = (args.session or "").strip()
    prepared = _prepare_runtime(restoring_session=bool(session_id))
    holder.runtime = prepared
    _write_pending_warnings(prepared, as_json=args.json)
    loaded, model_registry = _apply_active_model_selection(prepared.loaded, args.model)
    host = ChrysSessionHost(
        profile_name=args.agent,
        session_id=session_id or None,
        loaded_settings=loaded,
        model_registry=model_registry,
        approval_mode=ApprovalMode.BYPASS,
        cwd=cwd,
        on_successful_turn=on_buddy_successful_turn,
    )
    if model_registry is not None:
        # --model was applied host-locally (CLI provenance, no process pointer);
        # pin it so a settings reload cannot revert the run to the global default.
        host.engine.pin_model_profile()
    started = time.monotonic()
    try:
        if session_id:
            # The restore loads settings from the saved session's own root, and
            # the headless run stream never carries the bus warnings that load
            # publishes (``Warning`` is not a run event type) — so the restore
            # is driven here and the target root's additions are written before
            # the run. ``start()`` is idempotent; the run does not restore twice.
            await host.start()
            _write_warning_events(
                _restore_delta_warnings(host.engine.loaded_settings, prepared.pending_warnings),
                prepared.localizer,
                as_json=args.json,
            )
        result = await host.run_until_final(prompt)
        _write_result(result, as_json=args.json, duration=time.monotonic() - started)
        return 0
    finally:
        await host.shutdown()


def _exception_message(exc: BaseException) -> str:
    # KeyError's ``str()`` wraps the message in quotes; ``args[0]`` keeps it clean.
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc) or type(exc).__name__


def _localized_or_english(
    reference: MessageRef,
    *,
    as_json: bool,
    runtime: PreparedRuntime | None,
) -> str:
    if as_json or runtime is None:
        return format_message(reference)
    return runtime.localizer.render(reference)


def _exception_display_message(
    exc: AgentProfileNotFoundError | AmbiguousSessionIdError | SessionNotFoundError | ModelProfileNotFoundError,
    *,
    as_json: bool,
    runtime: PreparedRuntime | None,
) -> str:
    english = _exception_message(exc)
    if as_json or runtime is None or exc.display_message is None:
        return english
    return runtime.localizer.render(exc.display_message)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``chrys run``."""
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.prompt is None) == (args.task is None):
        parser.error("provide either a prompt or --task FILE, not both")
    holder = PreparedRuntimeHolder()
    try:
        return asyncio.run(run_command(args, holder))
    except TaskFileError as exc:
        _write_error(str(exc), as_json=args.json, code=exc.code)
        return 1
    except HeadlessRunError as exc:
        runtime = holder.runtime
        if args.json or runtime is None or exc.event.display_message is None:
            message = _exception_message(exc)
        else:
            message = runtime.localizer.render(exc.event.display_message)
        _write_error(
            message,
            as_json=args.json,
            code=exc.event.code or "headless_run_error",
            session_id=exc.event.session_id,
        )
        return 1
    except SessionNotFoundError as exc:
        _write_error(
            _exception_display_message(exc, as_json=args.json, runtime=holder.runtime),
            as_json=args.json,
            code="session_not_found",
        )
        return 1
    except AmbiguousSessionIdError as exc:
        _write_error(
            _exception_display_message(exc, as_json=args.json, runtime=holder.runtime),
            as_json=args.json,
            code="session_ambiguous",
        )
        return 1
    except AgentProfileNotFoundError as exc:
        _write_error(
            _exception_display_message(exc, as_json=args.json, runtime=holder.runtime),
            as_json=args.json,
            code="profile_not_found",
        )
        return 1
    except ModelProfileNotFoundError as exc:
        _write_error(
            _exception_display_message(exc, as_json=args.json, runtime=holder.runtime),
            as_json=args.json,
            code="model_profile_not_found",
        )
        return 1
    except TimeoutError:
        _write_error(
            _localized_or_english(_HEADLESS_RUN_TIMEOUT.bind(), as_json=args.json, runtime=holder.runtime),
            as_json=args.json,
            code="timeout",
        )
        return 124
    except KeyboardInterrupt:
        _write_error(
            _localized_or_english(_INTERRUPTED.bind(), as_json=args.json, runtime=holder.runtime),
            as_json=args.json,
            code="interrupted",
        )
        return 130
    except Exception as exc:
        _write_error(_exception_message(exc), as_json=args.json)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
