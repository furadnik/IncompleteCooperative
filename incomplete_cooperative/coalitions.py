"""A set of utility functions working with coalitions."""
from __future__ import annotations

from typing import Iterable

from .protocols import Game, Player


class Coalition:
    """A coalition representation."""

    def __init__(self, id: int) -> None:
        """Initialize Coalition."""
        self.id = id

    def __repr__(self) -> str:
        """Represent the coalition."""
        return f"Coalition({self.players})"

    @staticmethod
    def from_players(players: Iterable[Player]) -> Coalition:
        """Get coalition by players."""
        id = 0
        for player in set(players):
            id += 2**player
        return Coalition(id)

    @property
    def players(self) -> Iterable[Player]:
        """Get players of coalition."""
        coalition = self.id
        i = 0
        while coalition:
            if coalition & 1:
                yield i
            coalition >>= 1
            i += 1

    def __contains__(self, other: Coalition | Player) -> bool:
        """Get whether or not the coalition contains value."""
        if isinstance(other, Player):
            other = player_to_coalition(other)
        return bool(self.id & other.id == other.id)

    def __and__(self, other: Coalition | Player) -> Coalition:
        """Return the addition of coalitions."""
        if isinstance(other, Player):
            other = player_to_coalition(other)
        return Coalition(self.id & other.id)

    def __len__(self) -> int:
        """Get size of coalition."""
        s = 0
        coalition = self.id
        while coalition:
            s += coalition & 1
            coalition >>= 1
        return s

    def __eq__(self, other) -> bool:
        """Compare coalitions."""
        if isinstance(other, Coalition):
            return self.id == other.id
        elif isinstance(other, int):
            return self.players == [other]
        return False


def player_to_coalition(player: Player) -> Coalition:
    """Get a singleton containing only the player."""
    return Coalition(2**player)


def grand_coalition(game: Game) -> Coalition:
    """Get grand coalition."""
    return Coalition(2**game.number_of_players - 1)


def all_coalitions(game: Game) -> Iterable[Coalition]:
    """Get all possible coalitions of a game."""
    return map(Coalition, range(2**game.number_of_players))


def exclude_coalition(exclude: Coalition, coalitions: Iterable[Coalition]) -> Iterable[Coalition]:
    """Get coalitions that do not icnlude anyone from the `exclude` coalition."""
    return filter(lambda coalition: coalition & exclude == 0,
                  coalitions)
