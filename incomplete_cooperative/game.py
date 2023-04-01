"""An incomplete cooperative game representation."""
from __future__ import annotations

from typing import Any, Callable, Iterable, Literal

import numpy as np

from .coalitions import Coalition
from .protocols import BoundableIncompleteGame, Value, Values

Coalitions = Iterable[Coalition]
CoalitionPlayers = Iterable[int]


class IncompleteCooperativeGame:
    """Represent a game."""

    _values_is_known_index = 0
    _values_lower_index = 1
    _values_upper_index = 2

    def __init__(self, number_of_players: int,
                 bounds_computer: Callable[[BoundableIncompleteGame], None]) -> None:
        """Save basic game info."""
        self.number_of_players = number_of_players
        self._bounds_computer = bounds_computer
        self._values = np.zeros((2**self.number_of_players, 3), Value)
        self._init_values()

    def _filter_out_coalitions(self, values: np.ndarray[Any, np.dtype[Value]],
                               coalitions: Iterable[Coalition] | None) -> np.ndarray[Any, np.dtype[Value]]:
        """Filter out coalitions."""
        if coalitions is None:
            return values
        indices = np.fromiter(map(lambda x: x.id, coalitions), dtype=int)
        return values[indices]

    def get_value(self, coalition: Coalition) -> Value:
        """Get a value for coalition."""
        if not self.is_value_known(coalition):
            raise ValueError("Value is not known.")
        return self._values[coalition.id, self._values_lower_index]

    def get_values(self, coalitions: Iterable[Coalition] | None = None) -> np.ndarray[Any, np.dtype[Value]]:
        """Get values, or `False` if not known."""
        r = self.get_upper_bounds()
        coalitions_list = list(coalitions) if coalitions is not None else None
        if not np.all(self.get_known_values(coalitions_list)):
            raise ValueError("Not all requested values are known.")
        return self._filter_out_coalitions(r, coalitions_list)

    def set_value(self, value: Value | int, coalition: Coalition) -> None:
        """Set value of a coalition."""
        self._values[coalition, self._values_upper_index] = value
        self._values[coalition, self._values_lower_index] = value
        self._values[coalition, self._values_is_known_index] = 1

    def set_values(self, values: Values,
                   coalitions: Iterable[Coalition] | None = None) -> None:
        """Set multiple values."""
        if coalitions is not None:
            indices = np.fromiter(map(lambda x: x.id, coalitions), Value, np.size(values))
            self._values[indices, self._values_upper_index] = values
            self._values[indices, self._values_lower_index] = values
            self._values[indices, self._values_is_known_index] = 1
        else:
            self._values[:, self._values_upper_index] = values
            self._values[:, self._values_lower_index] = values
            self._values[:, self._values_is_known_index] = 1

    def _init_values(self) -> None:
        """Initialize values to all unknowns, except empty coalition."""
        self._values.fill(0)
        self.set_value(0, Coalition.from_players([]))  # empty coalition always has 0 value

    def get_upper_bound(self, coalition: Coalition) -> Value:
        """Get upper bound for a coalition."""
        return self._values[coalition.id, self._values_upper_index]

    def get_upper_bounds(self, coalitions: Iterable[Coalition] | None = None) -> Values:
        """Get upper bounds for coalitions."""
        return self._filter_out_coalitions(self._values[:, self._values_upper_index], coalitions)

    def get_lower_bound(self, coalition: Coalition) -> Value:
        """Get lower bound for a coalition."""
        return self._values[coalition.id, self._values_lower_index]

    def get_lower_bounds(self, coalitions: Iterable[Coalition] | None = None) -> Values:
        """Get upper bounds for coalitions."""
        return self._filter_out_coalitions(self._values[:, self._values_lower_index], coalitions)

    def get_interval(self, coalition: Coalition) -> np.ndarray[Literal[2], np.dtype[Value]]:
        """Get the interval, ie. both the upper and lower bounds."""
        return self._values[coalition.id, self._values_lower_index:self._values_upper_index]

    def get_intervals(self, coalitions: Iterable[Coalition]) -> np.ndarray[tuple[Any, Literal[2]], np.dtype[Value]]:
        """Get the intervals for (some) coalitions. Defaults to all."""
        r = self._values[:, self._values_lower_index:self._values_upper_index]
        return self._filter_out_coalitions(r, coalitions)

    def is_value_known(self, coalition: Coalition) -> bool:
        """Decide whether the value for a coalition is known."""
        return bool(self._values[coalition.id, self._values_is_known_index])

    def get_known_values(self, coalitions: Iterable[Coalition] | None = None) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """Get an iterable of `Value`s, or `None` if not known."""
        return self._filter_out_coalitions(self._values[:, self._values_is_known_index], coalitions) == 1

    def set_known_values(self, known_values: Iterable[Value], coalitions: Iterable[Coalition] | None = None) -> None:
        """Set known values to the selected. Drop other values."""
        self._init_values()
        return self.set_values(np.fromiter(known_values, Value), coalitions)

    def reveal_value(self, value: Value, coalition: Coalition) -> None:
        """Reveal a previously unknown value of coalition."""
        if self.get_value(coalition) is not None:
            raise ValueError("Value was already known.")

        self.set_value(value, coalition)

    def set_upper_bounds(self, values: Values,
                         coalitions: Iterable[Coalition] | None = None) -> None:
        """Set values of (some) coalitions the game. Defaults to all coalitions."""
        if coalitions is None:
            self._values[:, self._values_upper_index] = values
        else:
            indices = np.fromiter(map(lambda x: x.id, coalitions), int)
            self._values[indices, self._values_upper_index] = values

    def set_upper_bound(self, value: Value, coalition: Coalition) -> None:
        """Set value of a specific coalition."""
        if self.is_value_known(coalition):
            raise AttributeError("The selected coalition already has known value.")
        self._values[coalition, self._values_upper_index] = value

    def set_lower_bounds(self, values: Values,
                         coalitions: Iterable[Coalition] | None = None) -> None:
        """Set values of (some) coalitions the game. Defaults to all coalitions."""
        if coalitions is None:
            self._values[:, self._values_lower_index] = values
        else:
            indices = np.fromiter(map(lambda x: x.id, coalitions), int)
            self._values[indices, self._values_lower_index] = values

    def set_lower_bound(self, value: Value, coalition: Coalition) -> None:
        """Set value of a specific coalition."""
        if self.is_value_known(coalition):
            raise AttributeError("The selected coalition already has known value.")
        self._values[coalition, self._values_lower_index] = value

    def __eq__(self, other) -> bool:
        """Compare two games."""
        if not isinstance(other, IncompleteCooperativeGame):
            raise AttributeError("Cannot compare games with anything else than games.")
        return bool(np.all(self._values == other._values))

    def compute_bounds(self) -> None:
        """Recompute bounds given (potentially new) information."""
        self._bounds_computer(self)

    @property
    def full(self) -> bool:
        """Decide whether the game is fully known."""
        return bool(np.all(self.get_known_values()))
