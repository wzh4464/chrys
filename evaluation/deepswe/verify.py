#!/usr/bin/env python3
"""What a PACT campaign runs to accept a mission on a DeepSWE task, by language.

The long-horizon track delegates to a campaign only when the workspace has a
deterministic verification command (``pact.verify_command``); DeepSWE tasks carry
none, so without this every run ended at the repaired baseline and PACT never saw
the goal contract and plan that clarification had produced for it. The hidden
DeepSWE tests are not in the task image; these run the repository's own suite at
the base commit.
"""

from __future__ import annotations

VERIFY_BY_LANGUAGE: dict[str, str] = {
    "go": "go test ./...",
    "python": "python -m pytest -q -x -p no:cacheprovider",
    "typescript": "npm test --silent",
    "javascript": "npm test --silent",
    "rust": "cargo test -q",
}


def verify_command_for(language: str | None) -> str:
    """The verify command for a task language, or an empty string when there is none."""
    return VERIFY_BY_LANGUAGE.get(str(language or "").strip().lower(), "")
