"""Test the approximate SAM bounds."""
import cProfile
import pstats
import random
import timeit
from functools import partial
from os import getenv
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from incomplete_cooperative.bounds import (
    compute_bounds_superadditive,
    compute_bounds_superadditive_monotone_approx_cached)
from incomplete_cooperative.coalitions import (Coalition, all_coalitions,
                                               minimal_game_coalitions)
from incomplete_cooperative.functoolz import powerset
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.game_properties import is_superadditive
from incomplete_cooperative.generators import k_budget_generator
from incomplete_cooperative.norms import l1_norm

mpl.use('TkAgg')
NUM_PL = int(getenv("NUMBER_OF_PLAYERS") or 4)
NUM = int(getenv("NUMBER_OF_REPET") or 1)
STEP = int(getenv("STEP") or 0)
OUTPUT = Path(getenv("SAVE_PATH") or ".")
REPEATS = int(getenv("BOUND_REPEATS") or 10)
SEED = int(getenv("SEED") or 42)
BOUNDS = [compute_bounds_superadditive] + [partial(compute_bounds_superadditive_monotone_approx_cached, repetitions=i) for i in range(REPEATS)]


GAMES = [k_budget_generator(NUM_PL) for _ in range(NUM)]


def try_bounds():
    """Run a random game, revealing values and computing the bounds."""
    cum_norms = np.zeros(len(BOUNDS))
    for game in GAMES:
        min_game = list(minimal_game_coalitions(NUM_PL))
        incomplete_game = IncompleteCooperativeGame(NUM_PL)
        incomplete_game.set_known_values(game.get_values(min_game), min_game)

        # for step in range(min(STEP, 2**NUM_PL - len(min_game))):
        #     unknown = np.arange(2**NUM_PL)[np.logical_not(incomplete_game.are_values_known())]
        #     random_ch = np.random.choice(unknown)
        #     incomplete_game.reveal_value(game.get_value(Coalition(random_ch)), Coalition(random_ch))
        for repeat, bounds in enumerate(BOUNDS):
            incomplete_game._bounds_computer = bounds
            incomplete_game.compute_bounds()
            print(incomplete_game.get_upper_bounds())
            cum_norms[repeat] += float(l1_norm(incomplete_game))
    avg_norms = cum_norms / NUM
    plt.plot(avg_norms)
    plt.show()


if __name__ == '__main__':
    try_bounds()
