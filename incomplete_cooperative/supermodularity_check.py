"""Check if a game is supermodular."""
from .coalitions import (Coalition, all_coalitions, get_sub_coalitions,
                         grand_coalition)
from .protocols import Game, Player


def check_supermodularity(game: Game, tolerance: float = 1e-10) -> tuple[Coalition, Coalition, Player] | None:
    """Return the violating coalitions and player; or None if the game is supermodular.

    Includes tolerance for rounding errors.
    """
    for T in all_coalitions(game):
        for i in filter(lambda i: i not in T, grand_coalition(game).players):
            rhs = game.get_value(T | Coalition.from_players({i})) - game.get_value(T)
            for S in filter(lambda s: s != T, get_sub_coalitions(T)):
                lhs = game.get_value(S | Coalition.from_players({i})) - game.get_value(S)
                if lhs > rhs + tolerance:  # pragma: no cover
                    return T, S, i
    return None


def check_failed_diagnostics(game: Game, T: Coalition, S: Coalition, i: Player) -> None:  # pragma: no cover
    """Print information about the failed test."""
    print(T, S, i)
    print(game.get_value(S | Coalition.from_players([i])), game.get_value(S))
    print(game.get_value(T | Coalition.from_players([i])), game.get_value(T))
    print(game.get_value(S | Coalition.from_players([i])) - game.get_value(S),
          game.get_value(T | Coalition.from_players([i])) - game.get_value(T))
