# Copyright (c) 2026 Chrys. All rights reserved.

"""ask_user tool — allows the agent to ask the user questions.

The actual user interaction is handled by ``AskUserMiddleware`` (a
``FunctionMiddleware``), which intercepts calls to this tool, publishes
a ``QuestionToUser`` event, and waits for an ``AskUserResponse`` before
writing the result into the invocation context.  The tool function itself
is only reached as a fallback when no middleware is present (e.g. tests).
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import BeforeValidator, Field

from chrys.service.tools.kinds import KIND_ASK_USER, tool

_OPTIONS_DESCRIPTION = (
    'Suggested answers as a flat array of strings, for example ["Option A", "Option B"]. Pass the array directly; '
    "do not JSON-encode it or put an encoded array in one item. Do not use objects or wrapper keys. "
    "Combine a label and explanation in one string. Omit when there are no useful choices."
)
_OPTION_STRING_KEYS = ("label", "value", "content", "text", "title", "name", "description")
_OPTION_NEST_KEYS = ("item", "items", "option", "options", "choices", "children")
_MAX_OPTION_NORMALIZATION_DEPTH = 8
# Total node-visit budget across one options normalization: the depth cap
# alone does not bound work — shared references (a small graph of lists each
# referenced many times) amplify into exponentially many visits. On
# exhaustion deeper flattening yields nothing, so the literal-text and
# drop-option fallbacks engage instead of the walk running away.
_MAX_OPTION_NORMALIZATION_VISITS = 10_000
# ``json.loads`` runs BEFORE the visit budget can meter what it produced, so
# a stringified array must be length-gated up front: a multi-megabyte encoded
# list would otherwise materialize millions of elements only for the budget
# to throw them away. Past the gate the literal text survives as an option,
# same as any other undecodable string.
_MAX_STRINGIFIED_OPTIONS_CHARS = 100_000
_OPTION_FALLBACK_SKIP_KEYS = frozenset(
    {
        "header",
        "multiselect",
        "preview",
        "question",
        "questions",
    }
)


def _nonempty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    # A str subclass's methods are user code; work on a base-slot exact copy.
    text = (value if type(value) is str else str.__str__(value)).strip()
    return text or None


def _fallback_key_allowed(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.replace("_", "").lower()
    return normalized not in _OPTION_FALLBACK_SKIP_KEYS


def _decode_stringified_array(text: str, budget: list[int]) -> list[object] | None:
    if len(text) > _MAX_STRINGIFIED_OPTIONS_CHARS:
        return None
    if not (text.startswith("[") and text.endswith("]")):
        return None
    # Parse cost scales with the text, not with the one entry visit already
    # charged, so charge by size too: a stack of near-gate-sized arrays then
    # exhausts the budget after a few decodes instead of each paying full
    # price under the gate.
    budget[0] -= len(text) // 1024
    if budget[0] < 0:
        return None
    try:
        decoded = json.loads(text)
    except ValueError, RecursionError:
        # ValueError, not just JSONDecodeError: the int digit limit raises a
        # plain ValueError, and the literal text must survive as an option.
        return None
    return decoded if isinstance(decoded, list) else None


type _NormalizationMemo = dict[object, tuple[object, Any]]


def _strip_memoized(value: object, memo: _NormalizationMemo) -> str | None:
    """``_nonempty_string`` memoized by object identity.

    Stripping is O(len); a small list holding many references to one huge
    string would otherwise pay a full pass per reference — the strip runs
    once per distinct object, so total strip work is bounded by the
    caller's actually-materialized payload.
    """
    if not isinstance(value, str):
        return None
    # The entry RETAINS the source object: an id() key alone would go stale
    # when a temporary (e.g. a decoded array element) is freed and a later,
    # distinct string reuses its address. Retention keeps the id unique for
    # the memo's lifetime; the identity check makes the guard explicit.
    key = ("strip", id(value))
    entry = memo.get(key)
    if entry is not None and entry[0] is value:  # type: ignore[index]
        return entry[1]  # type: ignore[index]
    text = _nonempty_string(value)
    memo[key] = (value, text)
    return text


def _flatten_option(
    value: object,
    *,
    depth: int = 0,
    _budget: list[int] | None = None,
    _memo: _NormalizationMemo | None = None,
) -> list[str]:
    if _budget is None:
        _budget = [_MAX_OPTION_NORMALIZATION_VISITS]
    if _memo is None:
        _memo = {}
    _budget[0] -= 1
    if _budget[0] < 0:
        return []
    if isinstance(value, str):
        # Repeated references to one string object flatten identically at a
        # given depth (decode eligibility is depth-gated), so the expansion —
        # including any stringified-array decode — is computed and charged
        # once and shared references cost O(1).
        flat_key = ("flat", id(value), depth)
        entry = _memo.get(flat_key)
        # Entries retain the source object (see _strip_memoized) so a freed
        # temporary's reused address can never alias a distinct string.
        if entry is not None and entry[0] is value:  # type: ignore[index]
            cached = entry[1]  # type: ignore[index]
            # The cached expansion costs O(1) work here but its SIZE still
            # counts against the budget: shared references to one decoded
            # array must not emit unbounded copies of its options.
            _budget[0] -= len(cached)
            if _budget[0] < 0:
                return []
            return cached
        result = _flatten_option_text(value, depth=depth, _budget=_budget, _memo=_memo)
        _memo[flat_key] = (value, result)
        return result
    if depth >= _MAX_OPTION_NORMALIZATION_DEPTH:
        return []
    if type(value) is list:
        out: list[str] = []
        for item in value:
            if _budget[0] < 0:
                break
            out.extend(_flatten_option(item, depth=depth + 1, _budget=_budget, _memo=_memo))
        return out
    if type(value) is not dict:
        # Only exact containers are traversed: a subclass's iteration and
        # lookup hooks are user code that can raise or block mid-validation.
        return []

    entries: dict[str, object] = {}
    for key, item in value.items():
        if _budget[0] < 0:
            break
        _budget[0] -= 1
        if isinstance(key, str):
            entries[key if type(key) is str else str.__str__(key)] = item

    out: list[str] = []
    for key in _OPTION_STRING_KEYS:
        if text := _strip_memoized(entries.get(key), _memo):
            out.extend(_flatten_option(text, depth=depth + 1, _budget=_budget, _memo=_memo))
            break
    for key in _OPTION_NEST_KEYS:
        if _budget[0] < 0:
            break
        if key in entries:
            out.extend(_flatten_option(entries[key], depth=depth + 1, _budget=_budget, _memo=_memo))
    if out:
        return out

    for key, item in entries.items():
        if _budget[0] < 0:
            break
        if _fallback_key_allowed(key) and (text := _strip_memoized(item, _memo)):
            return _flatten_option(text, depth=depth + 1, _budget=_budget, _memo=_memo)
    return []


def _flatten_option_text(
    value: str,
    *,
    depth: int,
    _budget: list[int],
    _memo: _NormalizationMemo,
) -> list[str]:
    """The string arm of ``_flatten_option``, split out for memoization."""
    text = _strip_memoized(value, _memo)
    if not text:
        return []
    if depth < _MAX_OPTION_NORMALIZATION_DEPTH:
        decoded = _decode_stringified_array(text, _budget)
        if decoded == []:
            return []
        if decoded is not None:
            # A decoded array that flattens to no strings (e.g. "[1, 100]")
            # keeps the literal option text instead of dropping the option.
            return _flatten_option(decoded, depth=depth + 1, _budget=_budget, _memo=_memo) or [text]
    return [text]


def _coerce_options(value: object) -> object:
    if value is None:
        return None
    budget = [_MAX_OPTION_NORMALIZATION_VISITS]
    memo: _NormalizationMemo = {}
    if isinstance(value, str) or type(value) is dict:
        return _flatten_option(value, _budget=budget, _memo=memo) or None
    if type(value) is list:
        out: list[str] = []
        for item in value:
            if budget[0] < 0:
                break
            out.extend(_flatten_option(item, _budget=budget, _memo=memo))
        return out or None
    return value


@tool(kind=KIND_ASK_USER)
async def ask_user(
    question: Annotated[
        str,
        (
            "The full question to ask the user. Put the complete question here; do not send the question as "
            "intermediate assistant text before calling this tool. Ask exactly one question in each call; if you "
            "need several answers, call ask_user once for the most blocking question, wait for the response, then "
            "call it again for the next question. Plain text and Markdown are both supported."
        ),
    ],
    options: Annotated[
        list[str] | None, Field(description=_OPTIONS_DESCRIPTION), BeforeValidator(_coerce_options)
    ] = None,
) -> str:
    """Ask the user a question and wait for their response.

    Use this when you need clarification, confirmation, or additional
    information from the user before proceeding.  The question supports
    plain text or Markdown; use Markdown to structure longer questions with
    headings, lists, tables, code blocks, or other readable formatting.

    Always put the complete question in the ``question`` argument. Do not
    send the question as intermediate assistant text and then call this tool
    with only a short prompt.

    Prefer this tool when the user's request is to ask them a question, or
    when a question affects how the current task should continue. It gives the
    user a better interactive experience and keeps the task inside the tool
    loop. If you present choices, examples, categories, or alternatives, put
    each one in ``options`` as a string; do not leave ``options`` empty while
    listing choices in the question text. If the user asks you to stop asking
    questions or stop trying this tool, stop and wait instead of asking another
    blocking question. If the task is already complete and your question is
    only a follow-up, put that question in the final response instead of
    calling this tool.

    This tool asks exactly one question and takes a flat list of string
    options. Do not bundle several independent questions, checklists, or
    multi-part clarification requests into one call. If you need several
    answers, ask the most blocking question first, wait for the user's response,
    then call ``ask_user`` again for the next question. Do not use
    ``questions``, ``header``, ``multiSelect``, or option objects. Valid shapes
    are ``{"question": "..."}`` and
    ``{"question": "...", "options": ["A", "B"]}``. For multiple questions,
    call ``ask_user`` once per question.
    """
    # Fallback when AskUserMiddleware is not in the pipeline (e.g. direct call in tests).
    return f"[Question for user]: {question}"
