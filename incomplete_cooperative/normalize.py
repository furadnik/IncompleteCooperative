"""Normalize games."""
from typing import cast

from incomplete_cooperative.game import IncompleteCooperativeGame, Value


def normalize_game(game: IncompleteCooperativeGame) -> None:
    """Normalize a game. Must not be minimal."""
    singletons = map(lambda x: game.players_to_coalition([x]), range(game.number_of_players))
    for singleton in singletons:
        singleton_value = cast(Value, game.get_value(singleton))
        for coalition in filter(lambda x: x & singleton, game.coalitions):
            game.set_value(coalition, cast(Value, game.get_value(coalition)) - singleton_value)

    grand_coalition_value = game.get_value(game.grand_coalition)

    if not grand_coalition_value:
        return

    upper_bounds = game.upper_bounds
    lower_bounds = game.lower_bounds
    upper_bounds /= grand_coalition_value
    lower_bounds /= grand_coalition_value
