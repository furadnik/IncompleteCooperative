"""Generators of games."""
from random import randrange

from .coalitions import all_coalitions
from .game import IncompleteCooperativeGame
from .protocols import Game, GameGenerator


def factory_generator(number_of_players: int, owner: int | None = None) -> Game:
    """Generate a `factory` game."""
    owner = randrange(0, number_of_players) if owner is None else owner
    game = IncompleteCooperativeGame(number_of_players, lambda x: None)
    for coalition in all_coalitions(game):
        if owner not in coalition:
            game.set_value(0, coalition)
        else:
            game.set_value((len(coalition) - 1)**2, coalition)
    return game


GENERATORS: dict[str, GameGenerator] = {
    "factory": factory_generator,
}
