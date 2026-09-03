# Copyright (c) 2026 Chrys. All rights reserved.

"""Console entry point for the external Chrys-PACT ACP agent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import cast

import acp as acp_sdk

from chrys.foundation.config.settings_store import LoadedSettings
from chrys.foundation.config.warnings import settings_warning_events
from chrys.orchestration.startup import bootstrap_runtime
from chrys.pact.server import default_server
from chrys.service.profiles.agents.registry import AgentProfileRegistry


def build_parser() -> argparse.ArgumentParser:
    """Build the intentionally small v1 command line."""
    parser = argparse.ArgumentParser(
        prog="chrys-pact",
        description="Run one PACT Campaign as an ACP stdio agent.",
    )
    parser.add_argument("--agent", default="Code", help="Chrys agent profile used for every PACT role")
    verification = parser.add_mutually_exclusive_group(required=True)
    verification.add_argument("--verify", metavar="COMMAND", help="Deterministic verification command")
    verification.add_argument(
        "--verify-from-settings",
        action="store_true",
        help="Read the verification command from the pact.verify_command setting",
    )
    verification.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Explicitly permit PACT completion without a verification command",
    )
    return parser


def _configure_logging() -> None:
    """Keep ACP stdout exclusive to JSON-RPC frames."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    logging.basicConfig(handlers=[handler], level=logging.WARNING)


def _prepare_runtime() -> LoadedSettings:
    """Bootstrap once and retain a project-free base settings snapshot."""
    bootstrap = bootstrap_runtime(
        dotenv_override=True,
        configure_stdio=True,
        project_root=None,
    )
    for warning in (*bootstrap.warnings, *settings_warning_events(bootstrap.loaded)):
        sys.stderr.write(f"Warning: {warning.message}\n")
    return bootstrap.loaded


def _resolve_profile_name(selector: str) -> str:
    """Fail before ACP session creation when the role profile is unusable."""
    registry = AgentProfileRegistry()
    registry.load_all()
    profile = registry.resolve_selector(selector)
    if profile is None:
        raise ValueError(f"Agent profile not found: {selector}")
    if profile.acp is not None:
        raise ValueError("--agent must select an in-process Chrys profile.")
    return profile.name


async def run_command(args: argparse.Namespace) -> int:
    """Run the one-process ACP server."""
    verify_command = args.verify.strip() if args.verify is not None else None
    if args.verify is not None and not verify_command:
        raise ValueError("--verify must contain a non-empty command.")
    loaded_settings = _prepare_runtime()
    if args.verify_from_settings:
        # Fail closed. A campaign whose verify command silently resolved to
        # nothing would report work as done without ever checking it, which is
        # the one thing the governance layer exists to prevent.
        verify_command = loaded_settings.settings.pact_verify_command.strip()
        if not verify_command:
            raise ValueError(
                "--verify-from-settings requires pact.verify_command to be set "
                "(project .chrys/settings.yaml, user settings, or CHRYS_PACT_VERIFY_COMMAND)."
            )
    profile_name = _resolve_profile_name(args.agent)
    server = default_server(
        profile_name=profile_name,
        loaded_settings=loaded_settings,
        verify_command=verify_command,
        allow_unverified=args.allow_unverified,
    )
    try:
        await acp_sdk.run_agent(cast(acp_sdk.Agent, server), use_unstable_protocol=True)
    finally:
        await server.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Synchronous console-script entry point."""
    _configure_logging()
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run_command(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        sys.stderr.write(f"Error: {detail}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
