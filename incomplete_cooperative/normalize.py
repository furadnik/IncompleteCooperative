"""Normalize games."""
from typing import Any

import numpy as np

from incomplete_cooperative.coalitions import (all_coalitions, grand_coalition,
                                               player_to_coalition)
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.graph_game import GraphCooperativeGame
from incomplete_cooperative.protocols import Game, MutableGame, Value

# norm info is information about the game that is required to undo the normalization
# that is the value of the singletons, and the value of grand coalitions without the singletons
NormInfo = tuple[Value, np.ndarray[Any, np.dtype[Value]]]


def _get_norminfo(game: Game) -> NormInfo:
    """Gather norminfo from game."""
    singletons = map(lambda x: player_to_coalition(x), range(game.number_of_players))
    singleton_values = game.get_values(singletons)
    grand_coalition_value = game.get_value(grand_coalition(game)) - np.sum(singleton_values)
    return grand_coalition_value, singleton_values


def normalize_game(game: MutableGame | GraphCooperativeGame) -> NormInfo:
    """Normalize a game. Must not be minimal."""
    norm_info = _get_norminfo(game)
    if isinstance(game, GraphCooperativeGame):
        _normalize_graph_game(game)
    elif isinstance(game, IncompleteCooperativeGame):
        _normalize_icg(game)
    else:  # pragma: no cover
        raise TypeError("Unknown game type.")
    return norm_info


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


def denormalize_game(game: MutableGame | GraphCooperativeGame, normalization_info: NormInfo) -> None:
    """Reconstruct the original game from normalization information."""
    if isinstance(game, GraphCooperativeGame):
        _denormalize_graph_game(game, normalization_info)
        return

    grand_coalition_value, singleton_values = normalization_info
    for coalition in all_coalitions(game):
        value = game.get_value(coalition)
        value *= grand_coalition_value
        for i in coalition.players:
            value += singleton_values[i]
        game.set_value(value, coalition)


def _denormalize_graph_game(game: GraphCooperativeGame, normalization_info: NormInfo) -> None:
    """Reconstruct the original graph game from normalization information."""
    grand_coalition_value, _ = normalization_info
    game._graph_matrix *= grand_coalition_value
