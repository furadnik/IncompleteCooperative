"""Generators of games."""
from functools import partial
from random import randrange
from typing import Callable

import numpy as np

from .coalitions import all_coalitions
from .game import IncompleteCooperativeGame
from .graph_game import GraphCooperativeGame
from .protocols import Value


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


def graph_generator(number_of_players: int,
                    dist_fn: Callable[[tuple[int, int]], np.ndarray] = lambda x: np.random.rand(*x)
                    ) -> GraphCooperativeGame:
    """Generate a `factory` game."""
    game_matrix = dist_fn((number_of_players, number_of_players)).astype(Value)
    return GraphCooperativeGame(game_matrix)


_gen = np.random.default_rng()
GENERATORS: dict[str, Callable[[int], GraphCooperativeGame | IncompleteCooperativeGame]] = {
    "factory": factory_generator,
    "factory_fixed": partial(factory_generator, owner=0),
    "graph": graph_generator,
    "graph_tirangular": partial(graph_generator, dist_fn=partial(_gen.triangular, 0, 0.6, 1)),
    "graph_increasing": partial(graph_generator, dist_fn=partial(_gen.triangular, 0, 1, 1)),
    "graph_decreasing": partial(graph_generator, dist_fn=partial(_gen.triangular, 0, 0, 1)),
    "graph_poiss": partial(graph_generator, dist_fn=partial(_gen.poisson, 1)),
    # graph_beta_{alpha}_{beta} ; alpha, beta \in [1, 5]
    **{f"graph_beta_{alpha}_{beta}": partial(graph_generator, dist_fn=partial(_gen.beta, alpha, beta))
       for alpha in range(1, 6) for beta in range(1, 6)},
    "graph_03_03": partial(graph_generator, dist_fn=partial(_gen.beta, 0.3, 0.3)),
    "graph_poiss5": partial(graph_generator, dist_fn=partial(_gen.poisson, 5)),
}
