"""Validate properties of games."""
import numpy as np

from .coalition_ids import all_coalitions, sub_coalitions
from .protocols import Game


def is_superadditive(game: Game) -> bool:
    """Test that `game` is superadditive."""
    values = game.get_values()
    for U in all_coalitions(game.number_of_players):
        Ss = sub_coalitions(U, game.number_of_players)
        Ts = U - Ss
        if not np.all(values[Ss] + values[Ts] <= values[U]):
            return False
    return True


def is_monotone_decreasing(game: Game) -> bool:
    """Test that game values are monotone increasing."""
    values = game.get_values()
    for U in all_coalitions(game.number_of_players):
        Ss = sub_coalitions(U, game.number_of_players)
        if not np.all(values[Ss] >= values[U]):
            return False
    return True


def is_sam(game: Game) -> bool:
    """Test that a game is superadditive and monotone-decreasing."""
    return is_superadditive(game) and is_monotone_decreasing(game)
