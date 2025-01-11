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
from incomplete_cooperative.game_properties import is_sam, is_superadditive
from incomplete_cooperative.generators import xos
from incomplete_cooperative.normalize import normalize_game
from incomplete_cooperative.norms import l1_norm

mpl.use('TkAgg')
NUM_PL = int(getenv("NUMBER_OF_PLAYERS") or 4)
NUM = int(getenv("NUMBER_OF_REPET") or 1)
STEP = int(getenv("STEP") or 0)
OUTPUT = Path(getenv("SAVE_PATH") or ".")
REPEATS = int(getenv("BOUND_REPEATS") or 10)
SEED = int(getenv("SEED") or 42)
BOUNDS = [compute_bounds_superadditive] + [partial(compute_bounds_superadditive_monotone_approx_cached, repetitions=i) for i in range(REPEATS)]


for game in range(100):
    for gen in [
            xos,
            partial(xos, number_of_additive=2),
            partial(xos, number_of_additive=3),
            partial(xos, number_of_additive=12),
    ]:
        game = gen(5)
        assert is_sam(game), f"{game.get_values()} {gen}"
        normalize_game(game)
        assert is_superadditive(game), f"{game.get_values()} {gen}"
