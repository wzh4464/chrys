# Copyright (c) 2026 Chrys. All rights reserved.

"""Bilingual heuristic classification of a turn into confidence bands."""

from __future__ import annotations

import pytest

from chrys.service.profiles.agents.schema import LongHorizonConfig
from chrys.service.routing.classifier import (
    DEFAULT_BANDS,
    RouteBand,
    TurnPlan,
    band_for,
    extract_prompt_signals,
    plan_for,
    prompt_score,
)
from chrys.service.routing.readiness import WorkspaceReadiness

_STRONG_EN = (
    "Implement end-to-end OAuth login: add the provider abstraction, migrate the user table, "
    "update the API, write integration tests, and document the flow. Acceptance criteria: "
    "1) existing sessions keep working 2) new users can sign up with Google 3) all tests pass. "
    "Touch src/auth/provider.py, src/api/routes.py and web/src/login.tsx as needed."
)
_STRONG_ZH = (
    "重构整个支付模块，迁移到新的事件总线，涉及 orders、billing、notifications 三个子系统，并补齐回归测试。"  # noqa: RUF001
    "验收标准：1）现有订单流程不受影响 2）所有测试通过 3）新事件总线覆盖全部支付路径。"  # noqa: RUF001
    "需要改动 src/orders/service.py、src/billing/ledger.py 与 src/notifications/dispatch.py。"
)


@pytest.mark.parametrize(
    "text",
    [
        "fix the typo in README",
        "what does TurnCoordinator do?",
        "把这个函数改名为 foo",
        "thanks",
        "ok",
        "继续",
    ],
)
def test_short_or_read_only_prompts_are_strong_standard(text: str) -> None:
    score, _ = prompt_score(extract_prompt_signals(text))

    assert band_for(score) is RouteBand.STRONG_STANDARD


@pytest.mark.parametrize("text", [_STRONG_EN, _STRONG_ZH])
def test_broad_and_specified_prompts_are_strong_long_horizon(text: str) -> None:
    signals = extract_prompt_signals(text)

    assert signals.archetype == "mutating_broad"
    score, reason = prompt_score(signals)
    assert band_for(score) is RouteBand.STRONG_LONG_HORIZON
    assert reason


@pytest.mark.parametrize("text", ["refactor the entire auth system", "把整个鉴权系统重构一下"])
def test_ambitious_but_unspecific_prompts_land_in_the_uncertain_band(text: str) -> None:
    """Scope without a plan is exactly what the LLM tiebreaker is for."""
    score, _ = prompt_score(extract_prompt_signals(text))

    assert band_for(score) is RouteBand.UNCERTAIN


def test_band_edges_are_half_open_upwards() -> None:
    bands = DEFAULT_BANDS

    assert band_for(0.0) is RouteBand.STRONG_STANDARD
    assert band_for(bands.strong_standard_max) is RouteBand.LEAN_STANDARD
    assert band_for(bands.lean_standard_max) is RouteBand.UNCERTAIN
    assert band_for(bands.uncertain_max) is RouteBand.LEAN_LONG_HORIZON
    assert band_for(bands.lean_long_horizon_max) is RouteBand.STRONG_LONG_HORIZON
    assert band_for(1.0) is RouteBand.STRONG_LONG_HORIZON


def test_plan_grades_by_band_and_readiness_only_vetoes_pact() -> None:
    config = LongHorizonConfig()
    ready = WorkspaceReadiness(True, True, True, False)
    not_ready = WorkspaceReadiness(False, True, True, False)

    assert plan_for(RouteBand.STRONG_STANDARD, config, ready) == TurnPlan()
    assert plan_for(RouteBand.LEAN_STANDARD, config, ready) == TurnPlan()
    assert plan_for(RouteBand.UNCERTAIN, config, ready) == TurnPlan()
    assert plan_for(RouteBand.LEAN_LONG_HORIZON, config, ready) == TurnPlan(True, True, False)
    assert plan_for(RouteBand.STRONG_LONG_HORIZON, config, ready) == TurnPlan(True, True, True)
    assert plan_for(RouteBand.STRONG_LONG_HORIZON, config, not_ready) == TurnPlan(True, True, False)


def test_the_profile_can_switch_off_either_long_horizon_stage() -> None:
    ready = WorkspaceReadiness(True, True, True, False)

    plan = plan_for(RouteBand.STRONG_LONG_HORIZON, LongHorizonConfig(localization=False), ready)
    assert plan == TurnPlan(False, True, True)

    plan = plan_for(RouteBand.STRONG_LONG_HORIZON, LongHorizonConfig(clarification=False), ready)
    assert plan == TurnPlan(True, False, True)


def test_an_unavailable_pact_tool_also_vetoes_delegation() -> None:
    config = LongHorizonConfig()
    no_tool = WorkspaceReadiness(True, True, False, False)

    assert plan_for(RouteBand.STRONG_LONG_HORIZON, config, no_tool).pact is False


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------


def test_question_shapes_are_read_only_in_both_languages() -> None:
    assert extract_prompt_signals("what does this function do?").archetype == "read_only"
    assert extract_prompt_signals("这个函数是做什么的？").archetype == "read_only"  # noqa: RUF001
    assert extract_prompt_signals("Why did the build fail").archetype == "read_only"


def test_scope_words_only_count_beside_a_change_verb() -> None:
    """ "Explain everything" is a big question, not a big change."""
    explain = extract_prompt_signals("explain everything about the entire auth system")
    rewrite = extract_prompt_signals("rewrite the entire auth system")

    assert explain.scope_hits == ()
    assert rewrite.scope_hits


@pytest.mark.parametrize(
    "text",
    [
        "把结果导出到 CSV",
        "从配置里导入这些值",
        "输出全部日志",
    ],
)
def test_chinese_words_containing_a_verb_character_do_not_false_match(text: str) -> None:
    """``导出`` contains ``出``; a substring match would read it as a change verb."""
    signals = extract_prompt_signals(text)

    assert signals.archetype != "mutating_broad"
    assert band_for(prompt_score(signals)[0]) is not RouteBand.STRONG_LONG_HORIZON


def test_english_verbs_match_whole_words_only() -> None:
    """``implementation`` is a noun; matching it as ``implement`` inflates scope."""
    signals = extract_prompt_signals("describe the entire implementation of the parser")

    assert signals.scope_hits == ()


def test_step_markers_are_counted_in_both_languages() -> None:
    english = extract_prompt_signals("First add the model, 2) migrate the table, and then update the API")
    chinese = extract_prompt_signals("先加模型，然后迁移表，再更新 API")  # noqa: RUF001

    assert english.step_markers >= 3
    assert chinese.step_markers >= 3


def test_acceptance_criteria_are_recognised_in_both_languages() -> None:
    assert extract_prompt_signals("Acceptance criteria: all tests pass").acceptance_hits
    assert extract_prompt_signals("验收标准：所有测试通过").acceptance_hits  # noqa: RUF001


def test_path_mentions_count_files_and_modules() -> None:
    signals = extract_prompt_signals("touch src/a.py, src/b.py and tests/test_c.py")

    assert signals.path_mentions >= 3


def test_chinese_word_count_is_estimated_from_characters() -> None:
    """Chinese has no spaces, so a space-split count would read as trivial."""
    long_chinese = "重构" * 100

    assert extract_prompt_signals(long_chinese).word_count >= 80


def test_the_reason_names_the_signals_that_fired() -> None:
    _score, reason = prompt_score(extract_prompt_signals(_STRONG_EN))

    assert "scope" in reason
    assert "acceptance" in reason


def test_score_is_clamped_to_the_unit_interval() -> None:
    for text in ("", "thanks", _STRONG_EN, _STRONG_EN * 3):
        score, _ = prompt_score(extract_prompt_signals(text))
        assert 0.0 <= score <= 1.0


def test_empty_input_is_strong_standard() -> None:
    score, _ = prompt_score(extract_prompt_signals("   "))

    assert band_for(score) is RouteBand.STRONG_STANDARD
