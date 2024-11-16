"""Functional utilities."""
from itertools import chain, combinations
from typing import Iterable, TypeVar

T = TypeVar("T")


def powerset(iter: list[T]) -> Iterable[tuple[T, ...]]:
    """Generate the powerset of an iterable."""
    return chain.from_iterable(
        combinations(iter, r) for r in range(len(iter))
    )
