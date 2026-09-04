# Copyright (c) 2026 Chrys. All rights reserved.

"""The brief records what the prior-experience recall returned, or why nothing."""

from __future__ import annotations

from chrys.orchestration.engine.run.long_horizon import render_memory_prior


def test_a_recalled_prior_is_rendered_under_its_status() -> None:
    section = render_memory_prior(
        "recalled 120 chars for repo 'parser-kit' (4.2s)", "Canonical rules:\n- Map call sites."
    )

    assert section.startswith("## Prior experience from the team graph")
    assert "Recall: recalled 120 chars for repo 'parser-kit' (4.2s)" in section
    assert section.rstrip().endswith("- Map call sites.")
    assert "(none)" not in section


def test_an_empty_prior_still_names_the_reason() -> None:
    section = render_memory_prior("timeout after 45s (45.0s)", "")

    assert "Recall: timeout after 45s (45.0s)" in section
    assert section.rstrip().endswith("(none)")
