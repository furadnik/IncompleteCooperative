"""Normalize games."""
from typing import cast

from incomplete_cooperative.coalitions import (all_coalitions, grand_coalition,
                                               player_to_coalition)
from incomplete_cooperative.game import IncompleteCooperativeGame, Value


def normalize_game(game: IncompleteCooperativeGame) -> None:
    """Normalize a game. Must not be minimal."""
    singletons = map(lambda x: player_to_coalition(x), range(game.number_of_players))
    for singleton in singletons:
        singleton_value = cast(Value, game.get_value(singleton))
        for coalition in filter(lambda x: x & singleton, all_coalitions(game)):
            game.set_value(game.get_value(coalition) - singleton_value, coalition)

    grand_coalition_value = game.get_value(grand_coalition(game))

    if not grand_coalition_value:
        return

    upper_bounds = game.get_upper_bounds()
    lower_bounds = game.get_lower_bounds()
    upper_bounds /= grand_coalition_value
    lower_bounds /= grand_coalition_value
