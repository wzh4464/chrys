# Copyright (c) 2026 Chrys. All rights reserved.

"""The single guarded model call the router may make on an uncertain turn.

Every failure mode resolves the same way -- ``standard`` -- because the cost of
guessing wrong upward is a whole PACT campaign and the cost of guessing wrong
downward is one ordinary turn. The verdict carries *why* it failed so the event
stream can show a silent model rather than a confident "standard".
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chrys.kernel import Message
from chrys.service.llm.json_extract import json_object_candidates, repair_json_object_candidate
from chrys.service.llm.responses import get_final_response
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.routing.classifier import PromptSignals
from chrys.service.routing.guard import TiebreakerGuard

logger = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = 150
_DEFAULT_TIMEOUT_SECONDS = 5.0

_SYSTEM = (
    "You classify one developer request as either a short task or a long-horizon task.\n"
    "A long-horizon task changes several modules or subsystems, needs a plan with "
    "verification, and cannot plausibly be finished and checked in a single pass.\n"
    "A short task is a question, a fix, a rename, or a change confined to one area, "
    "however ambitiously it is phrased.\n"
    'Reply with ONLY a JSON object: {"long_horizon": true|false, "confidence": 0.0-1.0, '
    '"reason": "<= 12 words"}. No prose, no code fence.'
)


@dataclass(frozen=True, slots=True)
class TiebreakerVerdict:
    """One model verdict, or the reason there is not one."""

    long_horizon: bool
    confidence: float
    reason: str
    failure: str = ""
    """``""``, ``timeout``, ``malformed``, ``unavailable``, ``rate_limited``, ``circuit_open``."""


def _unavailable(failure: str, reason: str = "") -> TiebreakerVerdict:
    return TiebreakerVerdict(long_horizon=False, confidence=0.0, reason=reason, failure=failure)


class LlmRouteClassifier:
    """Ask a model to break a tie the heuristic could not."""

    def __init__(
        self,
        profile: ModelProfile,
        *,
        guard: TiebreakerGuard,
        session_id: str | None,
        parent_session_id: str | None,
        session_dir: Path | None,
        client: Any | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_output_tokens: int = _MAX_OUTPUT_TOKENS,
    ) -> None:
        self._profile = profile
        self._guard = guard
        self._session_id = session_id
        self._parent_session_id = parent_session_id
        self._session_dir = session_dir
        self._client = client
        self._chat_options: dict[str, Any] | None = None
        self._timeout = timeout_seconds
        self._max_output_tokens = max_output_tokens

    def _get_client(self) -> Any:
        """Create the client once; a caller may inject a session-shared one."""
        if self._client is None:
            from chrys.service.llm.clients import create_client

            self._client = create_client(
                self._profile,
                session_id=self._session_id,
                parent_session_id=self._parent_session_id,
                session_dir=self._session_dir,
                use_route_session_context=True,
            )
        return self._client

    def _options(self) -> dict[str, Any]:
        if self._chat_options is None:
            from chrys.service.profiles.models.options import effective_chat_options

            options = dict(effective_chat_options(self._profile) or {})
            # One line of JSON. Anything larger is a model ignoring the contract,
            # and paying for it would defeat the point of a cheap tiebreaker.
            options["max_output_tokens"] = self._max_output_tokens
            self._chat_options = options
        return self._chat_options

    async def classify(self, text: str, signals: PromptSignals) -> TiebreakerVerdict:
        """Return the model's verdict, or a typed failure to fall back on."""
        allowed, denial = self._guard.allow()
        if not allowed:
            return _unavailable(denial)
        messages = [
            Message("system", [_SYSTEM]),
            Message("user", [_render(text, signals)]),
        ]
        try:
            response = await asyncio.wait_for(
                get_final_response(
                    self._get_client(),
                    messages,
                    stream=False,
                    options=self._options(),
                    timeout=self._timeout,
                ),
                self._timeout,
            )
        except TimeoutError:
            self._guard.record_failure()
            return _unavailable("timeout")
        except Exception as exc:
            # A model lock mismatch, a missing key, or an unreachable endpoint
            # all arrive here; none of them should stop the turn.
            logger.debug("routing tiebreaker unavailable: %s", exc)
            self._guard.record_failure()
            return _unavailable("unavailable")
        verdict = _parse_verdict(response.text or "")
        if verdict is None:
            self._guard.record_failure()
            return _unavailable("malformed")
        self._guard.record_success()
        return verdict


def _render(text: str, signals: PromptSignals) -> str:
    """Show the model the request and what the heuristic already saw."""
    return (
        f"Request:\n{text}\n\n"
        "Heuristic signals (advisory):\n"
        f"- archetype: {signals.archetype}\n"
        f"- words: {signals.word_count}\n"
        f"- step markers: {signals.step_markers}\n"
        f"- scope words with a change verb: {', '.join(signals.scope_hits) or 'none'}\n"
        f"- acceptance criteria: {', '.join(signals.acceptance_hits) or 'none'}\n"
        f"- files or modules named: {signals.path_mentions}"
    )


def _parse_verdict(text: str) -> TiebreakerVerdict | None:
    for candidate in json_object_candidates(text):
        payload = _loads(repair_json_object_candidate(candidate))
        if payload is None:
            continue
        long_horizon = payload.get("long_horizon")
        confidence = payload.get("confidence")
        if not isinstance(long_horizon, bool):
            continue
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            continue
        reason = payload.get("reason")
        return TiebreakerVerdict(
            long_horizon=long_horizon,
            confidence=min(1.0, max(0.0, float(confidence))),
            reason=reason.strip()[:200] if isinstance(reason, str) else "",
        )
    return None


def _loads(candidate: str) -> dict[str, Any] | None:
    try:
        value = json.loads(candidate)
    except ValueError, RecursionError:
        return None
    return value if isinstance(value, dict) else None
