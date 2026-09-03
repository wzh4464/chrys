# Copyright (c) 2026 Chrys. All rights reserved.

"""Whether this workspace can actually carry a governed PACT campaign.

Readiness is a veto, never a score. The opencode router this design borrows
from folds a second repository dimension into its confidence by taking a
minimum; that is wrong here, because the long-horizon track's value is
governance and verification rather than parallelism. A repository with no
verify command does not make the task smaller -- it makes the campaign
impossible, and only the campaign.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TEST_DIRECTORY_NAMES = ("tests", "test", "spec", "specs", "__tests__")
_TEST_FILE_MARKERS = ("_test.", "test_", ".test.", ".spec.")
# Files that name an ecosystem. Their presence, not their contents, is what a
# turn's shape depends on, so the fingerprint stays stable across edits.
_MANIFEST_NAMES = (
    "pyproject.toml",
    "setup.py",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Gemfile",
    "composer.json",
    "CMakeLists.txt",
)
_GIT_STATUS_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class WorkspaceReadiness:
    """What the workspace offers a long-horizon turn."""

    verify_command_configured: bool
    has_tests: bool
    pact_tool_available: bool

    @property
    def pact_ready(self) -> bool:
        """Whether a campaign can both run and be verified.

        Both halves are required: without ``pact.verify_command`` the campaign
        cannot tell done from broken, and without the sub-agent tool the model
        has no way to hand the work over.
        """
        return self.verify_command_configured and self.pact_tool_available


def probe_workspace_readiness(cwd: str, *, verify_command: str, pact_tool_available: bool) -> WorkspaceReadiness:
    """Inspect *cwd* cheaply enough to run before every routed turn.

    One ``scandir`` and a settings read. Nothing here spawns a subprocess:
    this runs inside message admission, so anything slower would delay the
    turn the user just started.
    """
    return WorkspaceReadiness(
        verify_command_configured=bool(verify_command.strip()),
        has_tests=_has_tests(Path(cwd)),
        pact_tool_available=pact_tool_available,
    )


def _has_tests(root: Path) -> bool:
    try:
        entries = list(os.scandir(root))
    except OSError:
        return False
    for entry in entries:
        name = entry.name
        if entry.is_dir() and name in _TEST_DIRECTORY_NAMES:
            return True
        if entry.is_file() and any(marker in name for marker in _TEST_FILE_MARKERS):
            return True
    return False


def probe_git_dirty(cwd: str) -> bool | None:
    """Return whether the tree has uncommitted changes, or ``None`` if not a repo.

    Diagnostic only, and deliberately NOT part of ``probe_workspace_readiness``:
    a dirty tree vetoes nothing, and ``git status`` on a large repository is far
    too slow to run on the admission path for a value no decision reads.
    """
    root = Path(cwd)
    if not (root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_STATUS_TIMEOUT_SECONDS,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def workspace_fingerprint(cwd: str) -> str:
    """Identify the workspace's *shape* for multi-turn route inheritance.

    Deliberately coarse: top-level directory names plus which ecosystem
    manifests exist. Editing a file must not invalidate an inherited routing
    decision, but adding a whole subsystem should.
    """
    root = Path(cwd)
    digest = hashlib.sha256()
    try:
        entries = sorted(os.scandir(root), key=lambda entry: entry.name)
    except OSError:
        digest.update(b"<unreadable>")
        return digest.hexdigest()
    for entry in entries:
        name = entry.name
        if entry.is_dir():
            if not name.startswith("."):
                digest.update(f"d:{name}\n".encode())
        elif name in _MANIFEST_NAMES:
            digest.update(f"m:{name}\n".encode())
    return digest.hexdigest()
