"""An incomplete cooperative game representation."""
from __future__ import annotations
from typing import Callable

Coalition = list[int]
CoalitionCode = int
Value = int


class IncompleteCooperativeGame:
    """Represent a game."""

    def __init__(self, number_of_players: Value,
                 known_values: dict[Coalition, Value],
                 bounds_computer: Callable[[], None]) -> None:
        """Save basic game info."""

    def set_known_values(self, known_values: dict[Coalition, Value]) -> None:
        """Save known values."""

    def _coalition_to_repr(self, coalition: Coalition) -> int:
        """Turn a Coalition into a numeric representation."""

    def get_value(self, coalition: Coalition) -> Value:
        """Get a value for coalition."""

    def reveal_value(self, coalition: Coalition, value: Value) -> None:
        """Reveal a value of a coalition."""

    def get_bounds(self, coalition: Coalition) -> tuple[int, int]:
        """Get bounds for a coalition."""

    def compute_bounds(self) -> None:
        """Recompute bounds given (potentially new) information."""

    @property
    def number_of_exploralbe_coalitions(self) -> int:
        """Get the number of explorable coalitions.

        Those are coalitions that aren't singletons.
        """

    @property
    def known_values(self) -> list[bool]:
        """Get a list of bools for each coalition, saying whether or not its value is known."""
