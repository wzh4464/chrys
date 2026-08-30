# Copyright (c) 2026 Chrys. All rights reserved.

"""Utility helpers for the diff_view package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def loop_last[T](values: Iterable[T]) -> Iterable[tuple[bool, T]]:
    """Iterate and generate a tuple with a flag for last value."""
    iter_values = iter(values)
    try:
        previous_value = next(iter_values)
    except StopIteration:
        return
    for value in iter_values:
        yield False, previous_value
        previous_value = value
    yield True, previous_value


def fill_lists[T](list_a: list[T], list_b: list[T], fill_value: T) -> None:
    """Make two lists the same size by extending the smaller with a fill value."""
    a_length = len(list_a)
    b_length = len(list_b)
    if a_length > b_length:
        list_b.extend([fill_value] * (a_length - b_length))
    elif b_length > a_length:
        list_a.extend([fill_value] * (b_length - a_length))
