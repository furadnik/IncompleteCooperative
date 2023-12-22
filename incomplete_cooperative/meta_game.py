"""A metagame is a game whose values are the divergences of another game."""
from typing import Iterable, Self

import numpy as np

from .coalitions import all_coalitions, get_k_zero
from .protocols import (Coalition, Game, GapFunction, MutableIncompleteGame,
                        Value, Values)


class MetaGame:
    """Meta-Game of a game holds its divergences."""

    def __init__(self, game: Game, incomplete: MutableIncompleteGame,
                 divergence: GapFunction) -> None:
        """Store initial values of the metagame."""
        self.k_zero = list(get_k_zero(game))
        k_zero = set(self.k_zero)
        self.players = [coalition for coalition in all_coalitions(game) if coalition not in k_zero]
        self.game = game
        self._incomplete = incomplete
        self.divergence = divergence

    @property
    def number_of_players(self) -> int:
        """The number of players in the game."""
        return len(self.players)

    def get_values(self, coalitions: Iterable[Coalition] | None = None) -> Values:
        """Get values for (some) coalitions the game. Defaults to all coalitions."""
        coalitions = coalitions if coalitions is not None else all_coalitions(self)
        return np.fromiter((self.get_value(coalition) for coalition in coalitions),
                           dtype=Value)

    def get_value(self, coalition: Coalition) -> Value:
        """Get value for a specific coalition."""
        inner_coalitions = [self.players[i] for i in coalition.players] + self.k_zero
        self._incomplete.set_known_values(self.game.get_values(inner_coalitions), inner_coalitions)
        self._incomplete.compute_bounds()
        return self.divergence(self._incomplete)

    def copy(self) -> Self:
        """Return a deep copy of the game."""
        raise NotImplementedError("Doesn't support copy.")

    def __add__(self, other: Self) -> Self:
        """Add games of the same type."""
        raise NotImplementedError("Doesn't support addition.")
