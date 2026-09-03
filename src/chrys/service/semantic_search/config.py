# Copyright (c) 2026 Chrys. All rights reserved.

"""Configuration for the optional semantic-localization preflight."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SemanticSearchMode(StrEnum):
    """Localization execution modes."""

    OFF = "off"
    FALLBACK = "fallback"
    AUTO = "auto"
    LLM = "llm"


@dataclass(frozen=True, slots=True)
class SemanticSearchConfig:
    """Validated limits and behavior for one localization run."""

    mode: SemanticSearchMode = SemanticSearchMode.AUTO
    max_iterations: int = 20
    top_locations: int = 12
    timeout_seconds: float = 120.0
    max_tool_results: int = 20
    model_profile: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.mode, str) and not isinstance(self.mode, SemanticSearchMode):
            object.__setattr__(self, "mode", SemanticSearchMode(self.mode))
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.top_locations < 1:
            raise ValueError("top_locations must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_tool_results < 1:
            raise ValueError("max_tool_results must be positive")
