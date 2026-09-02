# Copyright (c) 2026 Chrys. All rights reserved.

"""Semantic code localization services used by the optional CLI preflight.

Localization stays outside Chrys' engine and session layers. When an LLM or
CodeGraph is unavailable, the bundled deterministic indexer and graph adapter
still produce a useful report.
"""

from .config import SemanticSearchConfig, SemanticSearchMode
from .models import LocalizationArtifact, LocalizationResult
from .pipeline import SemanticSearchError, localize_requirement

__all__ = [
    "LocalizationArtifact",
    "LocalizationResult",
    "SemanticSearchConfig",
    "SemanticSearchError",
    "SemanticSearchMode",
    "localize_requirement",
]
