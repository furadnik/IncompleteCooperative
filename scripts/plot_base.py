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
    "solve_greedy": "Online Greedy",
    "best_states": "Online Optimal",
    "solve_offline_greedy": "Offline Greedy",
    "expected_best_states": "Offline Optimal",
}


def get_colors(number: int) -> Iterator:
    """Get colors from CMAP."""
    return map(CMAP, map(lambda x: (x + 1) / (number + 2), range(number)))
