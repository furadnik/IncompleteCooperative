"""An incomplete cooperative game representation."""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Literal

import numpy as np

from .coalitions import Coalition
from .protocols import BoundableIncompleteGame, Value, ValueIn, Values

Coalitions = Iterable[Coalition]
CoalitionPlayers = Iterable[int]

LOGGER = logging.getLogger(__name__)


def _none_bounds(game: BoundableIncompleteGame) -> None:
    """Do nothing."""


class IncompleteCooperativeGame:
    """Represent a game."""

    _values_is_known_index = 0
    _values_lower_index = 1
    _values_upper_index = 2

    def __init__(self, number_of_players: int,
                 bounds_computer: Callable[[BoundableIncompleteGame], None] = _none_bounds) -> None:
        """Save basic game info.

        Arguments:
            number_of_players: The number of players in the game.
            bounds_computer: The function that _modifies_ the game with bounds.
        """
        self.number_of_players = number_of_players
        self._bounds_computer = bounds_computer
        self._values = np.zeros((2**self.number_of_players, 3), Value)
        self._init_values()

    def __repr__(self) -> str:  # pragma: no cover
        """Representation of icg."""
        return f"ICG({self.are_values_known()})"

    def __neg__(self) -> IncompleteCooperativeGame:
        """Return the ICG with negative values."""
        ret = self.copy()
        ret._values[:, self._values_lower_index] = -self._values[:, self._values_upper_index]
        ret._values[:, self._values_upper_index] = -self._values[:, self._values_lower_index]
        return ret

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
        if not np.all(self.are_values_known(coalitions_list)):
            raise ValueError("Not all requested values are known.")
        return self._filter_out_coalitions(r, coalitions_list)

    def set_value(self, value: ValueIn, coalition: Coalition) -> None:
        """Set value of a coalition."""
        self._values[coalition.id, self._values_upper_index] = value
        self._values[coalition.id, self._values_lower_index] = value
        self._values[coalition.id, self._values_is_known_index] = 1

    def unset_value(self, coalition: Coalition) -> None:
        """Set value of a coalition."""
        self._values[coalition.id, self._values_upper_index] = 0
        self._values[coalition.id, self._values_lower_index] = 0
        self._values[coalition.id, self._values_is_known_index] = 0

    def set_values(self, values: Values,
                   coalitions: Iterable[Coalition] | None = None) -> None:
        """Set multiple values."""
        if coalitions is not None:
            indices = np.fromiter(map(lambda x: x.id, coalitions), int, np.size(values))
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

    def compute_bounds(self) -> None:
        """Compute the bounds."""
        self._bounds_computer(self)

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

    def get_interval(self, coalition: Coalition) -> np.ndarray[tuple[Literal[2]], np.dtype[Value]]:
        """Get the interval, ie. both the upper and lower bounds."""
        return self._values[coalition.id, self._values_lower_index:self._values_upper_index + 1]  # type: ignore[return-value]

    def get_intervals(self, coalitions: Iterable[Coalition] | None = None
                      ) -> np.ndarray[tuple[Any, Literal[2]], np.dtype[Value]]:
        """Get the intervals for (some) coalitions. Defaults to all."""
        r = self._values[:, self._values_lower_index:self._values_upper_index + 1]
        return self._filter_out_coalitions(r, coalitions)

    def is_value_known(self, coalition: Coalition) -> bool:
        """Decide whether the value for a coalition is known."""
        return bool(self._values[coalition.id, self._values_is_known_index])

    def are_values_known(self, coalitions: Iterable[Coalition] | None = None) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """Get an iterable of `Value`s, or `None` if not known."""
        return self._filter_out_coalitions(self._values[:, self._values_is_known_index], coalitions) == 1

    def get_known_value(self, coalition: Coalition) -> Value | None:
        """Get a value for coalition."""
        return self._values[coalition.id, self._values_lower_index] if self.is_value_known(coalition) else None

    def get_known_values(self, coalitions: Iterable[Coalition] | None = None) -> np.ndarray[Any, np.dtype[Value]]:
        """Get values, or `False` if not known."""
        all_values = np.copy(self.get_upper_bounds())
        coalitions_list = list(coalitions) if coalitions is not None else None
        wanted_values = self._filter_out_coalitions(all_values, coalitions_list)
        np.place(wanted_values, np.invert(self.are_values_known(coalitions_list)), None)
        return wanted_values

    def set_known_values(self, known_values: Iterable[ValueIn], coalitions: Iterable[Coalition] | None = None) -> None:
        """Set known values to the selected. Drop other values."""
        self._init_values()
        return self.set_values(np.fromiter(known_values, Value), coalitions)

    def reveal_value(self, value: ValueIn, coalition: Coalition) -> None:
        """Reveal a previously unknown value of coalition."""
        assert not self.is_value_known(coalition)
        self.set_value(value, coalition)

    def unreveal_value(self, coalition: Coalition) -> None:
        """Reveal a previously unknown value of coalition."""
        assert self.is_value_known(coalition)
        self.unset_value(coalition)

    def _get_coalition_map(self, coalitions: Iterable[Coalition] | None,
                           count: int = -1) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """Get a boolean array of the coalitions."""
        if coalitions is None:
            return np.ones(len(self._values), np.bool_)
        r = np.zeros(len(self._values), np.bool_)
        coalitions = np.fromiter(map(lambda x: x.id, coalitions), int, count)
        r[coalitions] = True
        return r

    def set_upper_bounds(self, values: Values,
                         coalitions: Iterable[Coalition] | None = None) -> None:
        """Set values of (some) coalitions the game. Defaults to all coalitions."""
        if coalitions is not None:
            coalitions = list(coalitions)
            all_values = np.zeros(len(self._values), dtype=Value)
            all_values[np.fromiter(map(lambda x: x.id, coalitions), int, len(values))] = values
        else:
            all_values = values

        relevant_positions = np.invert(self.are_values_known()) * self._get_coalition_map(coalitions, len(values))
        np.copyto(self._values[:, self._values_upper_index], all_values, where=relevant_positions)

    def set_upper_bound(self, value: ValueIn, coalition: Coalition) -> None:
        """Set value of a specific coalition."""
        self._values[coalition.id, self._values_upper_index] = value

    def set_lower_bounds(self, values: Values,
                         coalitions: Iterable[Coalition] | None = None) -> None:
        """Set values of (some) coalitions the game. Defaults to all coalitions."""
        if coalitions is not None:
            coalitions = list(coalitions)
            all_values = np.zeros(len(self._values), dtype=Value)
            all_values[np.fromiter(map(lambda x: x.id, coalitions), int, len(values))] = values
        else:
            all_values = values

        relevant_positions = np.invert(self.are_values_known()) * self._get_coalition_map(coalitions, len(values))
        np.copyto(self._values[:, self._values_lower_index], all_values, where=relevant_positions)

    def set_lower_bound(self, value: ValueIn, coalition: Coalition) -> None:
        """Set value of a specific coalition."""
        self._values[coalition.id, self._values_lower_index] = value

    def __eq__(self, other) -> bool:
        """Compare two games."""
        if not isinstance(other, IncompleteCooperativeGame):
            raise AttributeError("Cannot compare games with anything else than games.")
        return bool(np.all(self._values == other._values))

    @property
    def full(self) -> bool:
        """Decide whether the game is fully known."""
        return bool(np.all(self.are_values_known()))

    def copy(self) -> IncompleteCooperativeGame:
        """Copy the game."""
        new = IncompleteCooperativeGame(self.number_of_players, self._bounds_computer)
        new._values = np.copy(self._values)
        return new

    def __add__(self, other: IncompleteCooperativeGame) -> IncompleteCooperativeGame:
        """Add games."""
        assert all([
            np.all(self.are_values_known()),
            np.all(other.are_values_known()),
            self.number_of_players == other.number_of_players,
            isinstance(other, IncompleteCooperativeGame)
        ]), f"Calling game addition on incompatible games: {self}, {other}."
        new = self.copy()
        new._values[:, 1:3] += other._values[:, 1:3]
        return new
