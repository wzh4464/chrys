# Copyright (c) 2026 Chrys. All rights reserved.

"""Weak exclusion anchors for compaction state propagation."""

from __future__ import annotations

import weakref
from dataclasses import dataclass

from chrys.kernel import (
    EXCLUDE_REASON_KEY,
    EXCLUDED_KEY,
    GROUP_ANNOTATION_KEY,
    GROUP_ID_KEY,
    SUMMARIZED_BY_SUMMARY_ID_KEY,
    SUMMARY_OF_GROUP_IDS_KEY,
    Content,
    Message,
    set_excluded,
)
from chrys.kernel.identity import ContentList

from .events import _COMPACTION_EXCLUDE_REASONS, _REASON_COMPRESSION, CachedSummary
from .groups import _content_signature, _message_call_ids
from .summaries import _make_summary_message


@dataclass(frozen=True)
class _ExclusionAnchor:
    """Persist a tool-loop exclusion onto the matching state message.

    Current-turn messages are rebuilt by the framework's streaming finalizers,
    so identity is exact for non-streaming but not sufficient for streaming.
    Structural fallback is intentionally limited to role + call_ids, and scans
    from the end so current-turn tool calls win over older reused call ids.

    Identity tiers hold WEAK references. Anchors may survive an interrupted
    run whose ``after_run`` never fired; a weak tier lives exactly as long
    as its object, so a collected list or content simply stops matching —
    a recycled address can never false-match, and the anchor never keeps
    its source alive.
    """

    reason: str | None
    role: str
    contents_ref: weakref.ref[ContentList] | None
    call_ids: tuple[str, ...] = ()
    content_signature: tuple[tuple[str, str | None, str | None, str], ...] = ()
    message_id: str | None = None
    content_refs: tuple[weakref.ref[Content], ...] = ()

    @classmethod
    def from_message(cls, msg: Message) -> _ExclusionAnchor:
        return cls(
            reason=msg.additional_properties.get(EXCLUDE_REASON_KEY),
            role=msg.role,
            contents_ref=weakref.ref(msg.contents),
            call_ids=_message_call_ids(msg),
            content_signature=_content_signature(msg),
            message_id=msg.message_id,
            content_refs=tuple(weakref.ref(c) for c in msg.contents),
        )

    def find_in(self, messages: list[Message], used_indices: set[int]) -> int | None:
        snapshot = self.contents_ref() if self.contents_ref is not None else None
        if snapshot is not None:
            for i, msg in enumerate(messages):
                if i not in used_indices and msg.contents is snapshot and self._matches_shape(msg):
                    return i
        # The wire hands clients per-call views: fresh wrapper + fresh
        # contents LIST, but the content OBJECTS are shared with the state
        # original — matching on their identity keeps identity-level
        # precision for anchors snapshotted from a view. A dead ref matches
        # nothing: ``ref() is c`` cannot hold for a live content.
        if self.content_refs:
            for i, msg in enumerate(messages):
                if (
                    i not in used_indices
                    and len(msg.contents) == len(self.content_refs)
                    and all(ref() is c for ref, c in zip(self.content_refs, msg.contents, strict=True))
                    and self._matches_shape(msg)
                ):
                    return i
        if self.message_id:
            for i, msg in enumerate(messages):
                if i not in used_indices and msg.message_id == self.message_id and self._matches_shape(msg):
                    return i
        if self.content_signature or self.call_ids:
            for i in range(len(messages) - 1, -1, -1):
                if i in used_indices:
                    continue
                msg = messages[i]
                if self._matches_shape(msg):
                    return i
        return None

    def _matches_shape(self, msg: Message) -> bool:
        """Return whether *msg* matches the stable structural anchor keys."""
        if msg.role != self.role:
            return False
        if self.call_ids and _message_call_ids(msg) != self.call_ids:
            return False
        return not self.content_signature or _content_signature(msg) == self.content_signature


class ExclusionLedger:
    """Own compaction caches and weak exclusion anchors."""

    def __init__(self) -> None:
        self._summary_cache: dict[str, CachedSummary] = {}
        self._removed_group_ids: set[str] = set()
        self._excluded_anchors: list[_ExclusionAnchor] = []

    def reset_stale_exclusions(self, messages: list[Message]) -> None:
        """Reset stale foreign exclusions while retaining owned state."""
        if self._summary_cache or self._removed_group_ids:
            self.reinject_cached_summaries(messages)
            # Reset _excluded only for messages NOT managed by our compaction
            for msg in messages:
                if not msg.additional_properties.get(EXCLUDED_KEY, False):
                    continue
                annotation = msg.additional_properties.get(GROUP_ANNOTATION_KEY)
                if isinstance(annotation, dict):
                    if SUMMARIZED_BY_SUMMARY_ID_KEY in annotation:
                        continue  # Keep excluded — our cached summary replaces it
                    # Keep excluded if the group was fully removed (Phase 2/3)
                    gid = annotation.get(GROUP_ID_KEY)
                    if gid and gid in self._removed_group_ids:
                        continue
                    # Keep excluded if it's a summary of a removed group
                    summary_of = annotation.get(SUMMARY_OF_GROUP_IDS_KEY)
                    if summary_of and any(g in self._removed_group_ids for g in summary_of):
                        continue
                # Preserve exclusions from prior-session compaction that
                # this strategy instance doesn't track in its caches.
                reason = msg.additional_properties.get(EXCLUDE_REASON_KEY)
                if reason in _COMPACTION_EXCLUDE_REASONS:
                    continue
                msg.additional_properties[EXCLUDED_KEY] = False
        else:
            for msg in messages:
                if msg.additional_properties.get(EXCLUDED_KEY, False):
                    # Preserve exclusions set by compaction in a prior
                    # session — the _exclude_reason annotation survives
                    # serialization and identifies legitimate compaction
                    # flags even when the strategy has no cache.
                    reason = msg.additional_properties.get(EXCLUDE_REASON_KEY)
                    if reason in _COMPACTION_EXCLUDE_REASONS:
                        continue
                    msg.additional_properties[EXCLUDED_KEY] = False

    def persist_to_state(self, state_messages: list[Message]) -> None:
        """Propagate tool-loop exclusion flags to state messages once."""
        if not self._excluded_anchors:
            return
        used_indices: set[int] = set()
        for anchor in self._excluded_anchors:
            idx = anchor.find_in(state_messages, used_indices)
            if idx is None:
                continue
            used_indices.add(idx)
            set_excluded(state_messages[idx], excluded=True, reason=anchor.reason)
        self._excluded_anchors = []

    def snapshot_retry_state(self) -> tuple[_ExclusionAnchor, ...]:
        """Capture the current consume-once anchor list."""
        return tuple(self._excluded_anchors)

    def restore_retry_state(self, anchors: tuple[_ExclusionAnchor, ...]) -> None:
        """Restore anchors captured at retry-attempt entry."""
        self._excluded_anchors = list(anchors)

    def snapshot_excluded_anchors(self, messages: list[Message]) -> None:
        """Snapshot non-compression exclusions for after-run propagation."""
        self._excluded_anchors = [
            _ExclusionAnchor.from_message(msg)
            for msg in messages
            if msg.additional_properties.get(EXCLUDED_KEY, False)
            and msg.additional_properties.get(EXCLUDE_REASON_KEY) != _REASON_COMPRESSION
        ]

    def invalidate_for_fold(self, fold_range: list[Message]) -> None:
        """Remove compaction cache entries for messages folded by compression.

        When a cross-turn compression folds messages that were previously
        compacted (Phase 1) or removed (Phase 2/3), the corresponding
        ``_summary_cache`` and ``_removed_group_ids`` entries become stale.
        """
        for msg in fold_range:
            annotation = msg.additional_properties.get(GROUP_ANNOTATION_KEY)
            if not isinstance(annotation, dict):
                continue
            gid = annotation.get(GROUP_ID_KEY)
            if gid:
                summary_id = f"tool_summary_{gid}"
                self._summary_cache.pop(summary_id, None)
                self._removed_group_ids.discard(gid)

    def reinject_cached_summaries(self, messages: list[Message]) -> None:
        """Re-inject cached tool summaries missing from a replayed list.

        This is normally unnecessary for the active tool loop because
        compaction mutates the loop's list in place. It remains
        useful for copied/replayed lists where excluded originals reference a
        cached summary via ``SUMMARIZED_BY_SUMMARY_ID_KEY`` but the summary
        message itself is absent.
        """
        if not self._summary_cache:
            return

        # Collect summary_ids referenced by messages in this list
        present_ids = {msg.message_id for msg in messages if msg.message_id}
        needed: dict[str, int] = {}

        for i, msg in enumerate(messages):
            annotation = msg.additional_properties.get(GROUP_ANNOTATION_KEY)
            if not isinstance(annotation, dict):
                continue
            sid = annotation.get(SUMMARIZED_BY_SUMMARY_ID_KEY)
            if sid and sid in self._summary_cache and sid not in present_ids and sid not in needed:
                needed[sid] = i

        # Insert in reverse order to preserve indices
        for sid in sorted(needed, key=lambda item: needed[item], reverse=True):
            cached = self._summary_cache[sid]
            summary_msg = _make_summary_message(cached.summary_text, sid, cached.original_ids, cached.group_id)
            messages.insert(needed[sid], summary_msg)
