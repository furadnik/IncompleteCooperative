"""Generators of games."""
from functools import partial
from typing import Callable
from random import randrange

from .coalitions import all_coalitions
from .game import IncompleteCooperativeGame
from .protocols import Game, GameGenerator


def _none_bounds_computer(self) -> None:  # pragma: nocover
    return None


def factory_generator(number_of_players: int, owner: int | None = None,
                      bounds_computer: Callable = _none_bounds_computer) -> Game:
    """Generate a `factory` game."""
    owner = randrange(0, number_of_players) if owner is None else owner
    game = IncompleteCooperativeGame(number_of_players, bounds_computer)
    for coalition in all_coalitions(game):
        if owner not in coalition:
            game.set_value(0, coalition)
        else:
            game.set_value((len(coalition) - 1), coalition)
    return game


GENERATORS: dict[str, GameGenerator] = {  # TODO: supported
    "factory": factory_generator,
    "factory_fixed": partial(factory_generator, owner=0),
}
