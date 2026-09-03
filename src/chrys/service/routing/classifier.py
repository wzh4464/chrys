# Copyright (c) 2026 Chrys. All rights reserved.

"""Bilingual heuristic classification of one prompt into a confidence band.

The heuristic exists so the common cases never cost a model call: a typo fix
and a fully specified cross-module migration are both obvious, and only the
band between them is worth asking a model about.

Weights are calibrated against ``tests/service/routing/fixtures/calibration.jsonl``.
Changing one means re-running that gate: mis-promoting a turn costs a whole PACT
campaign, so the gate demands high precision and zero false positives in the
strong band rather than high recall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from chrys.service.profiles.agents.schema import LongHorizonConfig
from chrys.service.routing.readiness import WorkspaceReadiness


class RouteTrack(StrEnum):
    """Which pass shape a turn runs."""

    STANDARD = "standard"
    LONG_HORIZON = "long_horizon"


class RouteBand(StrEnum):
    """How sure the classifier is, and therefore how much work is justified."""

    STRONG_STANDARD = "strong_standard"
    LEAN_STANDARD = "lean_standard"
    UNCERTAIN = "uncertain"
    """The only band that may spend an LLM tiebreaker."""
    LEAN_LONG_HORIZON = "lean_long_horizon"
    """Full clarification with parallel localization, but no PACT delegation."""
    STRONG_LONG_HORIZON = "strong_long_horizon"


@dataclass(frozen=True, slots=True)
class BandThresholds:
    """Lower edges are inclusive, upper edges exclusive."""

    strong_standard_max: float = 0.25
    lean_standard_max: float = 0.45
    uncertain_max: float = 0.70
    lean_long_horizon_max: float = 0.85


DEFAULT_BANDS = BandThresholds()

Archetype = Literal["read_only", "trivial", "mutating_narrow", "mutating_broad"]


@dataclass(frozen=True, slots=True)
class PromptSignals:
    """What the prompt's surface says about the size of the task."""

    word_count: int
    step_markers: int
    scope_hits: tuple[str, ...]
    """Scope words that co-occur with a change verb; scope alone means nothing."""
    acceptance_hits: tuple[str, ...]
    path_mentions: int
    question_like: bool
    archetype: Archetype


@dataclass(frozen=True, slots=True)
class TurnPlan:
    """Which long-horizon stages this turn runs."""

    localization: bool = False
    clarification: bool = False
    pact: bool = False


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """One routing verdict, and enough provenance to explain it."""

    track: RouteTrack
    band: RouteBand
    plan: TurnPlan
    reason: str
    confidence: float
    source: str
    """``override``, ``profile``, ``heuristic``, ``llm``, ``inherited`` or ``guard``.

    A plain string rather than a Literal: the value crosses into ``TurnRouted``
    and the trajectory payload, and pinning it here only moves the cast to
    every construction site.
    """
    prompt_score: float = 0.0
    decided_at: float = 0.0
    archetype: Archetype = "mutating_narrow"
    inherited_from_turn: int | None = None
    switched_to: str = ""
    tiebreaker_failure: str = ""


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

_SCOPE_WORDS_EN = (
    "all",
    "every",
    "everything",
    "everywhere",
    "across",
    "entire",
    "whole",
    "end-to-end",
    "end to end",
    "throughout",
)
_SCOPE_WORDS_ZH = ("整个", "所有", "全部", "全量", "跨", "端到端", "整体")

# Whole-word matching only: "implementation" is a noun, and reading it as
# "implement" inflates every design discussion into a migration.
_CHANGE_VERBS_EN = (
    "refactor",
    "migrate",
    "implement",
    "rewrite",
    "replace",
    "port",
    "redesign",
    "overhaul",
    "restructure",
    "consolidate",
    "extract",
    "modernize",
    "unify",
    "standardize",
)
# Multi-word verbs, matched as substrings: a word-boundary pattern per
# inflection would be more machinery than "clean up" is worth.
_CHANGE_PHRASES_EN = ("clean up", "cleans up", "cleaned up", "cleaning up", "tidy up", "sort out")
# Chinese has no word boundaries, so the list holds whole words and never bare
# characters: matching "出" would read 导出 ("export") as a change verb.
_CHANGE_VERBS_ZH = (
    "重构",
    "迁移",
    "实现",
    "重写",
    "替换",
    "改造",
    "改写",
    "补齐",
    "覆盖",
    "重做",
    "拆分",
    "统一",
    "清理",
    "整理",
    "规范化",
)

_ACCEPTANCE_EN = ("acceptance criteria", "acceptance test", "must pass", "definition of done", "success criteria")
_ACCEPTANCE_ZH = ("验收标准", "验收条件", "完成标准", "必须满足", "需满足")

_QUESTION_WORDS_EN = ("what", "why", "how", "when", "where", "which", "who", "explain", "describe", "does", "is", "are")
_QUESTION_MARKERS_ZH = ("是什么", "为什么", "怎么", "如何", "吗", "呢")

_STEP_MARKERS_EN = ("first", "then", "next", "after that", "finally", "afterwards", "lastly")
# Alternation, longest first: a bare 先 has to count (先…然后…再 is the most
# common Chinese step chain) without also firing inside 首先.
_STEP_MARKERS_ZH = re.compile(r"首先|然后|接着|最后|其次|并且|再|先")

# Fullwidth punctuation is spelled with escapes: these are deliberately the
# Chinese forms, and the ambiguous-character lint cannot tell that apart from
# a typo in an ASCII string.
_NUMBERED_STEP = re.compile("(?:^|[\\s,;\\uff0c\\uff1b])(\\d{1,2}[)\\uff09.\\u3001]|[-*\\u2022]\\s)")
_PATH_PATTERN = re.compile(
    r"[\w.\-]+(?:/[\w.\-]+)+|\b[\w\-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|rb|c|cc|cpp|h|kt|swift)\b"
)
_CJK = re.compile("[\\u3400-\\u9fff\\uf900-\\ufaff]")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# Scoring weights. Grouped here so a calibration change is one diff.
_WEIGHT_LENGTH_LONG = 0.20
_WEIGHT_LENGTH_MEDIUM = 0.10
_WEIGHT_STEPS = 0.15
_WEIGHT_SCOPE_BASE = 0.30
_WEIGHT_SCOPE_EXTRA = 0.05
_WEIGHT_SCOPE_CAP = 0.40
_WEIGHT_ACCEPTANCE = 0.15
_WEIGHT_PATHS = 0.15
_WEIGHT_BROAD = 0.20
_PENALTY_TRIVIAL = 0.40
_PENALTY_SHORT = 0.30

_LONG_WORDS = 80
_MEDIUM_WORDS = 40
_MIN_STEP_MARKERS = 3
_MIN_PATH_MENTIONS = 3
_TRIVIAL_WORDS = 8


def extract_prompt_signals(text: str) -> PromptSignals:
    """Read the surface features the score is built from."""
    stripped = text.strip()
    lowered = stripped.lower()
    word_count = _word_count(stripped)
    change_verbs = _change_verb_hits(stripped, lowered)
    scope_hits = _scope_hits(stripped, lowered) if change_verbs else ()
    question_like = _question_like(stripped, lowered, has_change_verb=bool(change_verbs))
    if question_like:
        archetype: Archetype = "read_only"
    elif word_count < _TRIVIAL_WORDS and not scope_hits:
        archetype = "trivial"
    elif scope_hits:
        archetype = "mutating_broad"
    else:
        archetype = "mutating_narrow"
    return PromptSignals(
        word_count=word_count,
        step_markers=_step_markers(stripped, lowered),
        scope_hits=scope_hits,
        acceptance_hits=_acceptance_hits(stripped, lowered),
        path_mentions=len(_PATH_PATTERN.findall(stripped)),
        question_like=question_like,
        archetype=archetype,
    )


def prompt_score(signals: PromptSignals) -> tuple[float, str]:
    """Return a score in ``[0, 1]`` and the signals that produced it."""
    score = 0.0
    fired: list[str] = []
    if signals.word_count >= _LONG_WORDS:
        score += _WEIGHT_LENGTH_LONG
        fired.append(f"length={signals.word_count}")
    elif signals.word_count >= _MEDIUM_WORDS:
        score += _WEIGHT_LENGTH_MEDIUM
        fired.append(f"length={signals.word_count}")
    if signals.step_markers >= _MIN_STEP_MARKERS:
        score += _WEIGHT_STEPS
        fired.append(f"steps={signals.step_markers}")
    if signals.scope_hits:
        extra = _WEIGHT_SCOPE_EXTRA * (len(signals.scope_hits) - 1)
        score += min(_WEIGHT_SCOPE_BASE + extra, _WEIGHT_SCOPE_CAP)
        fired.append("scope=" + "/".join(signals.scope_hits))
    if signals.acceptance_hits:
        score += _WEIGHT_ACCEPTANCE
        fired.append("acceptance=" + "/".join(signals.acceptance_hits))
    if signals.path_mentions >= _MIN_PATH_MENTIONS:
        score += _WEIGHT_PATHS
        fired.append(f"paths={signals.path_mentions}")
    if signals.archetype == "mutating_broad":
        score += _WEIGHT_BROAD
        fired.append("archetype=mutating_broad")
    elif signals.archetype in {"read_only", "trivial"}:
        score -= _PENALTY_TRIVIAL
        fired.append(f"archetype={signals.archetype}")
    # A short prompt is usually a follow-up -- but "rewrite the entire auth
    # system" is five words and is exactly the case the uncertain band exists
    # for, so breadth suppresses the brevity penalty rather than losing to it.
    if signals.archetype != "mutating_broad" and (signals.question_like or signals.word_count < _TRIVIAL_WORDS):
        score -= _PENALTY_SHORT
        fired.append("short-or-question")
    clamped = min(1.0, max(0.0, score))
    return clamped, "; ".join(fired) if fired else "no long-horizon signals"


def band_for(score: float, bands: BandThresholds = DEFAULT_BANDS) -> RouteBand:
    """Map a score onto its band; lower edges are inclusive."""
    if score < bands.strong_standard_max:
        return RouteBand.STRONG_STANDARD
    if score < bands.lean_standard_max:
        return RouteBand.LEAN_STANDARD
    if score < bands.uncertain_max:
        return RouteBand.UNCERTAIN
    if score < bands.lean_long_horizon_max:
        return RouteBand.LEAN_LONG_HORIZON
    return RouteBand.STRONG_LONG_HORIZON


def plan_for(band: RouteBand, config: LongHorizonConfig, readiness: WorkspaceReadiness) -> TurnPlan:
    """Grade the work by band, with readiness vetoing only the campaign."""
    if band not in {RouteBand.LEAN_LONG_HORIZON, RouteBand.STRONG_LONG_HORIZON}:
        return TurnPlan()
    return TurnPlan(
        localization=config.localization,
        clarification=config.clarification,
        pact=band is RouteBand.STRONG_LONG_HORIZON and readiness.pact_ready,
    )


# ---------------------------------------------------------------------------
# signal extraction
# ---------------------------------------------------------------------------


def _word_count(text: str) -> int:
    """Estimate words across both scripts.

    Chinese has no spaces, so splitting on whitespace reads a paragraph as one
    word and every long Chinese requirement as trivial. Two characters per word
    is the usual approximation.
    """
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN_WORD.findall(text))
    return latin + cjk // 2


def _change_verb_hits(text: str, lowered: str) -> tuple[str, ...]:
    hits = [verb for verb in _CHANGE_VERBS_EN if re.search(rf"\b{verb}(?:s|ed|ing)?\b", lowered)]
    hits.extend(phrase for phrase in _CHANGE_PHRASES_EN if phrase in lowered)
    hits.extend(verb for verb in _CHANGE_VERBS_ZH if verb in text)
    return tuple(hits)


def _scope_hits(text: str, lowered: str) -> tuple[str, ...]:
    hits = [word for word in _SCOPE_WORDS_EN if re.search(rf"\b{re.escape(word)}\b", lowered)]
    hits.extend(word for word in _SCOPE_WORDS_ZH if word in text)
    return tuple(hits)


def _acceptance_hits(text: str, lowered: str) -> tuple[str, ...]:
    hits = [phrase for phrase in _ACCEPTANCE_EN if phrase in lowered]
    hits.extend(phrase for phrase in _ACCEPTANCE_ZH if phrase in text)
    return tuple(hits)


def _step_markers(text: str, lowered: str) -> int:
    count = len(_NUMBERED_STEP.findall(text))
    count += sum(1 for marker in _STEP_MARKERS_EN if re.search(rf"\b{re.escape(marker)}\b", lowered))
    count += len(_STEP_MARKERS_ZH.findall(text))
    return count


def _question_like(text: str, lowered: str, *, has_change_verb: bool) -> bool:
    """A question asks about code; it does not ask for a change to it."""
    if has_change_verb:
        return False
    if text.endswith(("?", "\uff1f")):
        return True
    if any(marker in text for marker in _QUESTION_MARKERS_ZH):
        return True
    first = lowered.split(maxsplit=1)
    return bool(first) and first[0] in _QUESTION_WORDS_EN
