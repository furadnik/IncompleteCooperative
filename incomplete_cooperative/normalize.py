"""Normalize games."""
from incomplete_cooperative.coalitions import (all_coalitions, grand_coalition,
                                               player_to_coalition)
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.graph_game import GraphCooperativeGame
from incomplete_cooperative.protocols import MutableGame


def normalize_game(game: MutableGame | GraphCooperativeGame) -> None:
    """Normalize a game. Must not be minimal."""
    if isinstance(game, GraphCooperativeGame):
        _normalize_graph_game(game)
    elif isinstance(game, IncompleteCooperativeGame):
        _normalize_icg(game)
    else:  # pragma: no cover
        raise TypeError("Unknown game type.")


def _normalize_icg(game: IncompleteCooperativeGame) -> None:
    """Normalize an icg."""
    singletons = map(lambda x: player_to_coalition(x), range(game.number_of_players))
    for singleton in singletons:
        singleton_value = game.get_value(singleton)
        for coalition in filter(lambda x: x & singleton, all_coalitions(game)):
            game.set_value(game.get_value(coalition) - singleton_value, coalition)

    grand_coalition_value = game.get_value(grand_coalition(game))

    if not grand_coalition_value:
        return

    upper_bounds = game.get_upper_bounds()
    lower_bounds = game.get_lower_bounds()
    upper_bounds /= grand_coalition_value
    lower_bounds /= grand_coalition_value


def _normalize_graph_game(game: GraphCooperativeGame) -> None:
    """Normalize a graph game."""
    grand_coalition_value = game.get_value(grand_coalition(game))

    if not grand_coalition_value:
        return

    for i in range(game.number_of_players):
        for j in range(i, game.number_of_players):
            game._graph_matrix[j, i] = 0

    game._graph_matrix /= grand_coalition_value
