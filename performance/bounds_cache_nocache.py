"""Compare the time of cached vs non-cached bounds compuing."""
import timeit

import numpy as np

from incomplete_cooperative.bounds import (compute_bounds_superadditive,
                                           compute_bounds_superadditive_cached)
from incomplete_cooperative.coalitions import (Coalition,
                                               minimal_game_coalitions)
from incomplete_cooperative.generators import factory_generator

NUM_PL = 6
NUM = 1000


def try_computing_bounds(number_of_players, bounds_computer):
    """Run a random game, revealing values and computing the bounds."""
    min_game = list(minimal_game_coalitions(number_of_players))
    incomplete_game = factory_generator(number_of_players, bounds_computer=bounds_computer)
    game = factory_generator(number_of_players, bounds_computer=bounds_computer)
    incomplete_game.set_known_values(game.get_values(min_game), min_game)
    incomplete_game.compute_bounds()

    for step in range(2**number_of_players - len(min_game)):
        unknown = np.arange(2**number_of_players)[np.logical_not(incomplete_game.are_values_known())]
        random = np.random.choice(unknown)
        incomplete_game.reveal_value(game.get_value(Coalition(random)), Coalition(random))
        incomplete_game.compute_bounds()


print(timeit.timeit(lambda: try_computing_bounds(NUM_PL, compute_bounds_superadditive), number=NUM))
print(timeit.timeit(lambda: try_computing_bounds(NUM_PL, compute_bounds_superadditive_cached), number=NUM))
print(timeit.timeit(lambda: try_computing_bounds(NUM_PL, compute_bounds_superadditive), number=NUM))
