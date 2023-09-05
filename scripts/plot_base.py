"""Base stuff regarding plotting."""
from typing import Iterator

import matplotlib as mpl  # type: ignore
from matplotlib import colormaps

mpl.rcParams.update({
    'axes.labelsize': 16,
    'font.size': 16,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': [6.5, 5.5]
})

CMAP = colormaps["inferno"]
NAME_MAP = {
    "random_eval": "Random",
    "eval": "PPO",
    "solve_greedy": "Local Optimal",
    "best_states": "Global Optimal",
    "expected_best_states": "Best Average",
}


def get_colors(number: int) -> Iterator:
    """Get colors from CMAP."""
    if number == 1:
        return iter([CMAP(.5)])
    return map(CMAP, map(lambda x: x / (number - 1), range(number)))
