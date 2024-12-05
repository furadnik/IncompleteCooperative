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

from incomplete_cooperative.bounds import \
    compute_bounds_superadditive_monotone_approx_cached
from incomplete_cooperative.coalitions import (Coalition, all_coalitions,
                                               minimal_game_coalitions)
from incomplete_cooperative.functoolz import powerset
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.game_properties import is_superadditive
from incomplete_cooperative.norms import l1_norm

mpl.use('TkAgg')
NUM_PL = int(getenv("NUMBER_OF_PLAYERS") or 10)
NUM = int(getenv("NUMBER_OF_REPET") or 10)
STEP = int(getenv("STEP") or 10)
OUTPUT = Path(getenv("SAVE_PATH") or ".")
REPEATS = int(getenv("BOUND_REPEATS") or 10)
SEED = int(getenv("SEED") or 42)
BOUNDS = [partial(compute_bounds_superadditive_monotone_approx_cached, repetitions=2**i) for i in range(11)]


def covg_fn_generator(bounds) -> IncompleteCooperativeGame:
    """Generate a set coverage fn."""
    universum = range(2 * NUM_PL)
    sets = [random.choice(list(powerset(list(universum)))) for _ in range(NUM_PL)]
    game = IncompleteCooperativeGame(NUM_PL, bounds)
    for coalition in all_coalitions(game):
        uni = set()
        for i in coalition.players:
            uni = uni.union(sets[i])
        game.set_value(-len(uni), coalition)
    assert is_superadditive(game)
    return game


def try_bounds(number_of_players, bounds) -> float:
    """Run a random game, revealing values and computing the bounds."""
    cum = 0.0
    np.random.seed(SEED)
    for _ in range(NUM):
        min_game = list(minimal_game_coalitions(number_of_players))
        incomplete_game = IncompleteCooperativeGame(number_of_players,
                                                    bounds_computer=bounds)
        game = covg_fn_generator(bounds)
        incomplete_game.set_known_values(game.get_values(min_game), min_game)
        incomplete_game.compute_bounds()

        for step in range(min(STEP, 2**number_of_players - len(min_game))):
            unknown = np.arange(2**number_of_players)[np.logical_not(incomplete_game.are_values_known())]
            random_ch = np.random.choice(unknown)
            incomplete_game.reveal_value(game.get_value(Coalition(random_ch)), Coalition(random_ch))
        incomplete_game.compute_bounds()
        cum += float(l1_norm(incomplete_game))
    return cum / NUM


def plot_bounds_different_repeats() -> None:
    """Plot bounds for different number of repeats."""
    avg_norms = []
    for repeat in range(REPEATS):
        avg_norms.append(try_bounds(number_of_players=NUM_PL, bounds=partial(
            compute_bounds_superadditive_monotone_approx_cached, repetitions=repeat)))
        print(avg_norms)
        plt.plot(avg_norms)
        plt.show()


if __name__ == '__main__':
    plot_bounds_different_repeats()
