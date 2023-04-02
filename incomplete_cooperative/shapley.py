"""A computer of the Shapley value."""
from itertools import starmap
from math import factorial
from typing import Any, Iterator

import numpy as np

from .coalitions import (Coalition, all_coalitions, exclude_coalition,
                         player_to_coalition)
from .protocols import Game, Value


def _get_contributions(number_of_players: int) -> np.ndarray[Any, np.dtype[Value]]:
    """Get contribution coeficient (not devided by n!) for each coalition size."""
    return np.fromiter((factorial(s) * factorial(number_of_players - s - 1) for s in range(number_of_players)),
                       Value, number_of_players)


def compute_shapley_value(game: Game) -> Iterator[Value]:
    """Compute the Shapley value."""
    coalition_contribution_coefficients = _get_contributions(game.number_of_players)
    n_fac = factorial(game.number_of_players)
    for singleton in map(player_to_coalition, range(game.number_of_players)):
        coalitions_without_player = np.fromiter(
            exclude_coalition(singleton, all_coalitions(game)),
            Coalition, 2**(game.number_of_players - 1))

        coalitions_with_player = np.fromiter(
            map(lambda coalition: coalition | singleton,
                coalitions_without_player),
            Coalition, 2**(game.number_of_players - 1))

        values_without = game.get_values(coalitions_without_player)
        values_with = game.get_values(coalitions_with_player)

        coalition_contributions = coalition_contribution_coefficients[list(map(
            len, coalitions_without_player))]

        yield sum(starmap(lambda cont, val_with, val_wo: cont * (val_with - val_wo),
                          zip(coalition_contributions, values_with, values_without))) / n_fac
