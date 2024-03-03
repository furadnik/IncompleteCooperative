"""Generators of games."""
from functools import partial
from math import exp
from typing import Callable, Protocol

import numpy as np
from networkx import Graph  # type: ignore[import]
from networkx import (adjacency_matrix, connected_watts_strogatz_graph,
                      geographical_threshold_graph, gnp_random_graph,
                      random_geometric_graph, random_internet_as_graph)

from .coalitions import all_coalitions
from .game import IncompleteCooperativeGame
from .graph_game import GraphCooperativeGame
from .protocols import Value

_gen = np.random.default_rng()

GeneratorFn = Callable[[int, np.random.Generator], GraphCooperativeGame | IncompleteCooperativeGame]


class GraphGen(Protocol):
    """Protocol for graph generators."""

    def __call__(self, _n: int, seed: int | np.random.Generator | None = None) -> Graph:
        """Call generates the graph."""


def _none_bounds_computer(self) -> None:  # pragma: nocover
    return None


def _fac_sq_fn(val: int) -> int:  # pragma: nocover
    return val**2


def _fac_one_fn(val: int) -> int:  # pragma: nocover
    return 1


def factory_generator(number_of_players: int,
                      generator: np.random.Generator = np.random.default_rng(), *,
                      owner: int | None = None,
                      bounds_computer: Callable = _none_bounds_computer,
                      value_fn: Callable = lambda x: x,
                      random_weights: bool = False
                      ) -> IncompleteCooperativeGame:
    """Generate a `factory` game."""
    owner = int(generator.integers(number_of_players) if owner is None else owner)  # nosec
    weights = generator.uniform(high=10, size=(number_of_players,)) if random_weights else np.ones(number_of_players)
    weights[owner] = 0
    game = IncompleteCooperativeGame(number_of_players, bounds_computer)
    for coalition in all_coalitions(game):
        if owner not in coalition:
            game.set_value(0, coalition)
        else:
            game.set_value(value_fn(np.sum(weights[list(coalition.players)])), coalition)
    return game


def factory_cheerleader_generator(number_of_players: int,
                                  generator: np.random.Generator = np.random.default_rng(), *,
                                  owner: int | None = None, cheerleader: int | None = None,
                                  bounds_computer: Callable = _none_bounds_computer) -> IncompleteCooperativeGame:
    """Generate a `factory` game with a cheerleader."""
    owner = int(generator.integers(number_of_players) if owner is None else owner)  # nosec
    while cheerleader is None or cheerleader == owner:  # pragma: no cover
        cheerleader = generator.integers(number_of_players)  # nosec

    game = IncompleteCooperativeGame(number_of_players, bounds_computer)
    for coalition in all_coalitions(game):
        coalition_value = 3 * (len(coalition) - 2) if cheerleader in coalition else len(coalition) - 1
        if owner not in coalition:
            game.set_value(0, coalition)
        else:
            game.set_value(coalition_value, coalition)
    return game


def factory_cheerleader_next_generator(number_of_players: int,
                                       generator: np.random.Generator = np.random.default_rng(), *,
                                       bounds_computer: Callable = _none_bounds_computer
                                       ) -> IncompleteCooperativeGame:
    """Generate a `factory` game with a cheerleader, who is owner+1."""
    owner = int(generator.integers(number_of_players))  # nosec
    cheerleader = (owner + 1) % number_of_players
    return factory_cheerleader_generator(number_of_players, generator, owner=owner,
                                         cheerleader=cheerleader, bounds_computer=bounds_computer)


def graph_generator(number_of_players: int,
                    generator: np.random.Generator = np.random.default_rng(), *,
                    dist_fn: Callable[[tuple[int, int]], np.ndarray] = lambda x: _gen.random(x)
                    ) -> GraphCooperativeGame:
    """Generate a graph game.

    Note: For now, this ignores the `generator`, assumes a generator is present in the `dist_fn`.
    """
    game_matrix = dist_fn((number_of_players, number_of_players)).astype(Value)
    return GraphCooperativeGame(game_matrix)


def convex_generator(number_of_players: int,
                     generator: np.random.Generator = np.random.default_rng()) -> IncompleteCooperativeGame:
    """Generate a `convex` game.

    Doesn't implement the generator seeding.
    """
    import pyfmtools as fmp  # type: ignore[import]

    # generate the coalition values using the pyfmtools library
    env = fmp.fm_init(number_of_players)
    while True:
        size, values = fmp.generate_fmconvex_tsort(1, number_of_players, number_of_players - 1, 1000, 1, 1000, env)
        values_2bit = fmp.ConvertCard2Bit(values, env)
        if fmp.IsMeasureSupermodular(values_2bit, env):
            break
    fmp.fm_free(env)

    # convert the values to a game
    game = IncompleteCooperativeGame(number_of_players)
    game.set_values(values_2bit, all_coalitions(game))
    return game


_LAST_OWNER = 0


def predictible_factory_generator(number_of_players: int, generator: np.random.Generator = np.random.default_rng()
                                  ) -> IncompleteCooperativeGame:
    """Generate a factory, choose the owner predictibly."""
    global _LAST_OWNER
    _LAST_OWNER = (_LAST_OWNER + 1) % number_of_players
    return factory_generator(number_of_players, owner=_LAST_OWNER)


def graph_to_game(graph: Graph, number_of_players: int | None = None) -> GraphCooperativeGame:
    """Turn a `Graph` to a graph game.

    Restrict the number of players, if supplied.
    """
    graph_array = adjacency_matrix(graph).toarray().astype(Value)
    if number_of_players is not None:
        graph_array = graph_array[:number_of_players, :number_of_players]
    return GraphCooperativeGame(graph_array)


def graph_gen_to_game(number_of_players: int, generator: np.random.Generator = np.random.default_rng(),
                      graph_gen: GraphGen = partial(gnp_random_graph, p=0.5)) -> GraphCooperativeGame:
    """Generate a graph cooperative game from a game generator, taking a seed and a number of nodes."""
    return graph_to_game(graph_gen(number_of_players, seed=generator),
                         number_of_players=number_of_players)  # nosec


def cycle(number_of_players: int, generator: np.random.Generator = np.random.default_rng()) -> GraphCooperativeGame:
    """Generate a graph game where the players are all on a random cycle."""
    permutation = generator.permutation(number_of_players)
    graph_array = np.zeros((number_of_players, number_of_players), dtype=Value)
    graph_array[permutation, np.roll(permutation, 1)] = 1
    graph_array[permutation, np.roll(permutation, -1)] = 1
    return GraphCooperativeGame(graph_array)


def additive(number_of_players: int, generator: np.random.Generator = np.random.default_rng(),
             weights_dist_fn: Callable[[np.random.Generator], float] = np.random.Generator.random
             ) -> IncompleteCooperativeGame:
    """Generate a random additive game, based on the distribution function."""
    game_state = np.zeros(2**number_of_players)
    for i in range(number_of_players):
        singleton = weights_dist_fn(generator)
        game_state[np.arange(2**number_of_players) & 2**i != 0] += singleton
    ig = IncompleteCooperativeGame(number_of_players)
    ig.set_values(game_state)
    return ig


def xos(number_of_players: int, generator: np.random.Generator = np.random.default_rng(), number_of_additive: int = 6,
        additive_gen: GeneratorFn = additive, normalize: bool = True,
        normalize_additive: bool = False) -> IncompleteCooperativeGame:
    """Generate a random OSX game out of `number_of_additive` additive games.

    It uses `additive_gen` to generate the additive games.
    `normalize` normalizes the grand coalition of the resulting game to 1.
    `normalize_additive` normalizes all the generated additive games BEFORE doing the XOR
    (reaching "less random" distribution of the games, presumably).
    """
    additive_values = [additive_gen(number_of_players, generator).get_values()
                       for _ in range(number_of_additive)]
    if normalize_additive:
        for value in additive_values:
            value /= value[-1]
    osx_values = np.max(np.array(additive_values), axis=0)
    ig = IncompleteCooperativeGame(number_of_players)
    if normalize:
        osx_values = osx_values / osx_values[-1]
    ig.set_values(osx_values)
    return ig


GENERATORS: dict[str, GeneratorFn] = {
    "factory": factory_generator,
    "predictible_factory": predictible_factory_generator,
    "factory_one": partial(factory_generator, value_fn=_fac_one_fn),
    "factory_square": partial(factory_generator, value_fn=_fac_sq_fn),
    "factory_exp": partial(factory_generator, value_fn=exp),
    "factory_fixed": partial(factory_generator, owner=0),
    "factory_cheerleader": factory_cheerleader_generator,
    "factory_cheerleader_next": factory_cheerleader_next_generator,
    "noisy_factory": partial(factory_generator, random_weights=True),
    "noisy_factory_square": partial(factory_generator, value_fn=_fac_sq_fn, random_weights=True),
    "noisy_factory_exp": partial(factory_generator, value_fn=exp, random_weights=True),
    "noisy_factory_fixed": partial(factory_generator, owner=0, random_weights=True),
    "noisy_factory_cheerleader": partial(factory_cheerleader_generator, random_weights=True),
    "noisy_factory_cheerleader_next": partial(factory_cheerleader_next_generator, random_weights=True),
    "graph": graph_generator,
    "graph_tirangular": partial(graph_generator, dist_fn=partial(_gen.triangular, 0, 0.6, 1)),
    "graph_increasing": partial(graph_generator, dist_fn=partial(_gen.triangular, 0, 1, 1)),
    "graph_decreasing": partial(graph_generator, dist_fn=partial(_gen.triangular, 0, 0, 1)),
    # graph_beta_{alpha}_{beta} ; alpha, beta \in [1, 5]
    **{f"graph_beta_{alpha}_{beta}": partial(graph_generator, dist_fn=partial(_gen.beta, alpha, beta))
       for alpha in range(1, 6) for beta in range(1, 6)},
    "graph_03_03": partial(graph_generator, dist_fn=partial(_gen.beta, 0.3, 0.3)),
    **{f"graph_poiss_{lam}": partial(graph_generator, dist_fn=partial(_gen.poisson, lam))
       for lam in [0.1, 0.01, 0.5, 1, 5, 10, 50]},
    "convex": convex_generator,
    **{_key: partial(graph_gen_to_game, graph_gen=gen) for _key, gen in {
        "graph_random": partial(gnp_random_graph, p=0.5),
        "graph_ws_connected": partial(connected_watts_strogatz_graph, k=3, p=0.5),
        "graph_internet": random_internet_as_graph,
        "graph_geometric": partial(random_geometric_graph, radius=1),
        "graph_geographical_treshold": partial(geographical_threshold_graph, theta=0.1),
    }.items()},
    "graph_cycle": cycle,
    "xos": xos,
}
