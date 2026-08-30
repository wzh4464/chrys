# Copyright (c) 2026 Chrys. All rights reserved.

"""Helpers for mutation provenance values displayed by the TUI."""

from __future__ import annotations

from chrys.service.mutations.types import MutationSource


def merge_mutation_source(current: str, source: str) -> str:
    """Merge mutation provenance while preserving any implicit marker."""
    if current == MutationSource.IMPLICIT.value or source == MutationSource.IMPLICIT.value:
        return MutationSource.IMPLICIT.value
    return source or current
