"""Generators of games."""
from functools import partial
from random import randrange
from typing import Callable

import numpy as np

from .coalitions import all_coalitions
from .game import IncompleteCooperativeGame
from .graph_game import GraphCooperativeGame
from .protocols import GameGenerator, Value


def _none_bounds_computer(self) -> None:  # pragma: nocover
    return None


def factory_generator(number_of_players: int, owner: int | None = None,
                      bounds_computer: Callable = _none_bounds_computer) -> IncompleteCooperativeGame:
    """Generate a `factory` game."""
    owner = randrange(0, number_of_players) if owner is None else owner  # nosec
    game = IncompleteCooperativeGame(number_of_players, bounds_computer)
    for coalition in all_coalitions(game):
        if owner not in coalition:
            game.set_value(0, coalition)
        else:
            game.set_value((len(coalition) - 1), coalition)
    return game


def graph_generator(number_of_players: int) -> GraphCooperativeGame:
    """Generate a `factory` game."""
    game_matrix = np.random.rand(number_of_players, number_of_players).astype(Value)
    return GraphCooperativeGame(game_matrix)


GENERATORS: dict[str, GameGenerator] = {
    "factory": factory_generator,
    "graph": graph_generator,
    "factory_fixed": partial(factory_generator, owner=0),
}
