"""A set of utility functions working with coalitions."""
from __future__ import annotations

from itertools import chain, combinations
from typing import Iterable, TypeVar

from .protocols import Game, IncompleteGame, Player

T = TypeVar("T")


def powerset(input_iter: Iterable[T]) -> Iterable[list[T]]:
    """Return the powerset of the input."""
    input_iter = list(input_iter)
    return map(list, chain.from_iterable(combinations(input_iter, i) for i in range(len(input_iter) + 1)))


class Coalition:
    """A coalition representation."""

    def __init__(self, id: int) -> None:
        """Initialize Coalition."""
        self.id = id

    def __repr__(self) -> str:  # pragma: no cover
        """Represent the coalition."""
        return f"Coalition({list(self.players)})"

    def __hash__(self) -> int:
        """Hash coalition."""
        return hash(self.id)

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

    def __or__(self, other: Coalition | Player) -> Coalition:
        """Return the addition of coalitions."""
        if isinstance(other, Player):
            other = player_to_coalition(other)
        return Coalition(self.id | other.id)

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

    def __sub__(self, other) -> Coalition:
        """Subtract coalition."""
        if isinstance(other, Coalition):
            return Coalition(self.id & ~other.id)
        if isinstance(other, int):
            return Coalition(self.id & ~(2**other))
        else:  # pragma: nocover
            raise ValueError(f"Cannot subtract Coalition and {other}")

    def __add__(self, other) -> Coalition:
        """Add a player."""
        if isinstance(other, int):
            return Coalition(self.id | 2**other)
        else:  # pragma: nocover
            raise ValueError(f"Cannot add Coalition and {other}")

    def inverted(self, number_of_players: int) -> Coalition:
        """Get the inverted coalition."""
        return grand_coalition(number_of_players) - self


def player_to_coalition(player: Player) -> Coalition:
    """Get a singleton containing only the player."""
    return Coalition(2**player)


def grand_coalition(players: Game | int) -> Coalition:
    """Get grand coalition."""
    if isinstance(players, Game):
        players = players.number_of_players
    return Coalition(2**players - 1)


def all_coalitions(players: Game | int) -> Iterable[Coalition]:
    """Get all possible coalitions of a game."""
    if isinstance(players, Game):
        players = players.number_of_players
    return map(Coalition, range(2**players))


def minimal_game_coalitions(players: Game | int) -> Iterable[Coalition]:
    """Get minimal value."""
    yield Coalition(0)
    yield grand_coalition(players)
    number_of_players = players if isinstance(players, int) else players.number_of_players
    yield from (Coalition.from_players([i]) for i in range(number_of_players))


def exclude_coalition(exclude: Coalition, coalitions: Iterable[Coalition]) -> Iterable[Coalition]:
    """Get coalitions that do not icnlude anyone from the `exclude` coalition."""
    return filter(lambda coalition: not len(coalition & exclude),
                  coalitions)


def get_known_coalitions(game: IncompleteGame) -> Iterable[Coalition]:
    """Get the coalitions that are known."""
    return (x for x in all_coalitions(game) if game.is_value_known(x))


def sub_coalitions(coalition: Coalition) -> Iterable[Coalition]:
    """Return all sub_coalitions of coalition."""
    return map(Coalition.from_players, powerset(coalition.players))


def disjoint_coalitions(coalition1: Coalition, coalition2: Coalition) -> bool:
    """Get whether or not two coalitions are disjoint."""
    return (coalition1 & coalition2) == Coalition(0)
