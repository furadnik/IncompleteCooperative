"""A computer of the Shapley value."""
from .game import IncompleteCooperativeGame, Coalition, Value
from typing import Callable
from math import factorial


def _defalut_game_get_value(game: IncompleteCooperativeGame, coalition: Coalition) -> Value:
    """Implement default `get_value` function."""
    return game.get_value(coalition)


def _weight_of_contribution(number_of_players: int, size_of_coalition: int) -> float:
    """Return the weight of the contribution."""
    return (
        factorial(size_of_coalition) * factorial(number_of_players - size_of_coalition - 1)
    ) / factorial(number_of_players)


def compute_shapley_value(game: IncompleteCooperativeGame,
                          get_value: Callable = _defalut_game_get_value) -> None:  # TODO: typehints.
    """Compute the Shapley value.

    The `get_value` function must get a `IncompleteCooperativeGame` and a `Coalition` and return its value.
    It is done like this to allow flexibility -- sometimes, we want to compute the Shapley value from up/low bounds.
    """
    for player in range(game.number_of_players):
        player_singleton = game.players_to_coalition([player])
        coalitions_without_player = game.filter_coalitions_not_include_coalition(player_singleton, game.coalitions)

        result = 0
        for coalition_without_player in coalitions_without_player:
            coalition_size = game.get_coalition_size(coalition_without_player)
            coalition_with_player = coalition_without_player & player_singleton
            value_with_player = get_value(game, coalition_with_player)
            value_without_player = get_value(game, coalition_without_player)
            contribution = value_with_player - value_without_player
            result += _weight_of_contribution(game.number_of_players, coalition_size) * contribution

        yield result
