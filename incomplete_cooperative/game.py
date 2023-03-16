"""An incomplete cooperative game representation."""
from __future__ import annotations
import numpy as np
from typing import Callable, Iterable

Coalition = int
Coalitions = Iterable[Coalition]
CoalitionPlayers = Iterable[int]
Value = np.float32


class IncompleteCooperativeGame:
    """Represent a game."""

    _values_is_known_index = 0
    _values_lower_index = 1
    _values_upper_index = 2

    def __init__(self, number_of_players: Value,
                 bounds_computer: Callable[[IncompleteCooperativeGame], None],
                 known_values: dict[Coalition, Value] = None) -> None:
        """Save basic game info."""
        self.number_of_players = number_of_players
        self._bounds_computer = bounds_computer
        self._values = np.zeros((2**self.number_of_players, 3), np.float32)
        self._init_values()

        if known_values:
            self.set_known_values(known_values)

    def _init_values(self) -> None:
        """Initialize values to all unknowns, except empty coalition."""
        self._values.fill(0)
        self.set_value(self.players_to_coalition([]), 0)  # empty coalition always has 0 value

    def __eq__(self, other: IncompleteCooperativeGame) -> None:
        """Compare two games."""
        return np.all(self._values == other._values)

    @property
    def coalitions(self) -> Coalitions:
        """Get all coalitions."""
        return range(2**self.number_of_players)

    def filter_coalitions_not_include_coalition(self, include: Coalition, coalitions: Coalitions) -> Coalitions:
        """Allow only those from `coalitions`, that do not include any players from `include`."""
        return filter(lambda coal: include & coal == 0, coalitions)

    def filter_coalitions_include_some_coalition(self, include: Coalition, coalitions: Coalitions) -> Coalitions:
        """Allow only those from `coalitions`, that include at least some players from `include`."""
        return filter(lambda coal: include & coal != 0, coalitions)

    def filter_coalitions_include_coalition(self, include: Coalition, coalitions: Coalitions) -> Coalitions:
        """Allow only those from `coalitions`, that include all players from `include`."""
        return filter(lambda coal: include & coal == include and include != 0,
                      coalitions)

    def filter_coalition_subset(self, coalition: Coalition, coalitions: Coalitions, proper: bool = False) -> Coalitions:
        """Allow only coalitions that are a (proper?) subset of `coalition`."""
        return filter(lambda coal: coalition & coal == coalition and (not proper or coal - coalition > 0),
                      coalitions)

    def get_coalition_size(self, coalition: Coalition) -> int:
        """Get size of a coalition."""
        s = 0
        while coalition:
            s += coalition & 1
            coalition >>= 1
        return s

    def set_known_values(self, known_values: dict[Coalition, Value]) -> None:
        """Save known values."""
        self._init_values()
        for coalition, value in known_values.items():
            self.set_value(coalition, value)

    def set_value(self, coalition: Coalition, value: Value) -> None:
        """Set value of a coalition."""
        self._values[coalition, self._values_upper_index] = value
        self._values[coalition, self._values_lower_index] = value
        self._values[coalition, self._values_is_known_index] = 1  # the value is known

    def get_value(self, coalition: Coalition) -> Value | None:
        """Get a value for coalition."""
        if not self.is_value_known(coalition):
            return None
        return self._values[coalition, self._values_lower_index]

    def players_to_coalition(self, players: CoalitionPlayers) -> Coalition:
        """Turn a Coalition into a numeric representation."""
        coalition = set(players)
        if coalition and max(coalition) >= self.number_of_players:
            raise AttributeError("This game doesn't have enough players for this.")
        return sum(map(lambda x: 2**x, coalition))

    def coalition_to_players(self, coalition: Coalition) -> CoalitionPlayers:
        """Turn the numeric representation to list of players."""
        if coalition >= 2**self.number_of_players:
            raise AttributeError("This coalition is too large for this game.")

        i = 0
        while coalition > 0:
            if coalition & 1:
                yield i
            coalition >>= 1
            i += 1

    def reveal_value(self, coalition: Coalition, value: Value) -> None:
        """Reveal a value of a coalition."""
        if self.get_value(coalition) is not None:
            raise ValueError("Value was already known.")

        self.set_value(coalition, value)

    def get_lower_bound(self, coalition: Coalition) -> Value:
        """Get lower bound for a coalition."""
        return self._values[coalition, self._values_lower_index]

    def get_upper_bound(self, coalition: Coalition) -> Value:
        """Get upper bound for a coalition."""
        return self._values[coalition, self._values_upper_index]

    def set_upper_bound(self, coalition: Coalition, bound: Value) -> None:
        """Set upper bound of a coalition."""
        if self.is_value_known(coalition):
            raise AttributeError("The selected coalition already has known value.")
        self._values[coalition, self._values_upper_index] = bound

    def set_lower_bound(self, coalition: Coalition, bound: Value) -> None:
        """Set lower bound of a coalition."""
        if self.is_value_known(coalition):
            raise AttributeError("The selected coalition already has known value.")
        self._values[coalition, self._values_lower_index] = bound

    def compute_bounds(self) -> None:
        """Recompute bounds given (potentially new) information."""
        self._bounds_computer(self)

    def is_value_known(self, coalition: Coalition) -> bool:
        """Is value of the coalition already known."""
        return bool(self._values[coalition, self._values_is_known_index])

    @property
    def known_values(self) -> list[bool]:
        """Get a list of bools for each coalition, saying whether or not its value is known."""
        return self._values[:, self._values_is_known_index] == 1

    @property
    def full(self) -> bool:
        """Decide whether the game is fully known."""
        return np.all(self.known_values)

    @property
    def upper_bounds(self) -> list[Value]:
        """Get all upper bounds."""
        return self._values[:, self._values_upper_index]

    @property
    def lower_bounds(self) -> list[Value]:
        """Get all lower bounds."""
        return self._values[:, self._values_lower_index]

    @property
    def values(self) -> list[Value]:
        """Get values, or `False` if not known."""
        return self.upper_bounds * self.known_values

    def get_bounds(self, coalition: Coalition) -> tuple[Value, Value]:
        """Get bounds for a coalition."""
        return self._values[coalition, self._values_lower_index:self._values_upper_index + 1]

    @property
    def grand_coalition(self) -> Coalition:
        """Get the grand coalition."""
        return 2**self.number_of_players - 1
