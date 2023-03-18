"""A computer of the Shapley value."""
from math import factorial
from typing import Any, Callable, Iterable

import numpy as np

from .game import Coalition, IncompleteCooperativeGame, Value


def _get_contributions(number_of_players: int) -> np.ndarray[Any, np.dtype[Value]]:
    """Get contribution coeficient (not devided by n!) for each coalition size."""
    return np.array([factorial(s) * factorial(number_of_players - s - 1) for s in range(number_of_players)])


def compute_shapley_value(game: IncompleteCooperativeGame,
                          get_values: Callable = lambda x: x.values,
                          get_values_without: Callable | None = None) -> Iterable[Value]:
    """Compute the Shapley value.

    The `get_values` function must get a `IncompleteCooperativeGame` and a `Coalition` and return its value.
    It is done like this to allow flexibility -- sometimes, we want to compute the Shapley value from up/low bounds.
    Finally, the `get_values_without` is a function that will compute the values of coalitions without the player.
    It defaults to `get_values` if not specified.
    """
    if get_values_without is None:
        get_values_without = get_values

    coalition_contribution_coefficients = _get_contributions(game.number_of_players)
    n_fac = factorial(game.number_of_players)
    for player in range(game.number_of_players):
        player_singleton = game.players_to_coalition([player])
        coalitions_without_player = np.fromiter(game.get_coalitions_not_including_players([player]), Coalition)

        values_without = get_values_without(game)[coalitions_without_player]
        values_with = get_values(game)[coalitions_without_player | player_singleton]
        coalition_contributions = coalition_contribution_coefficients[list(map(game.get_coalition_size,
                                                                               coalitions_without_player))]

        yield np.sum(coalition_contributions * (values_with - values_without)) / n_fac
