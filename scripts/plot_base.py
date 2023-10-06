"""Base stuff regarding plotting."""
import sys
from typing import Iterator

import matplotlib as mpl  # type: ignore
from matplotlib import colormaps
from matplotlib.colors import Colormap

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
    "solve_greedy": "Oracle Greedy",
    "best_states": "Oracle Optimal",
    "expected_greedy": "Offline Greedy",
    "expected_best_states": "Offline Optimal",
}
LINE_STYLES = ["solid", "dashed", "dashdot", "dotted"]

MULTIFIG_SIZES = (8.3, 9.5)
MULTIFIG_RECT = [0, 0.035, 1, 1]


def get_colors(number: int) -> Iterator[tuple[Colormap, str]]:
    """Get colors from CMAP.

    Returns an iterator of tuples:
        the color as a color map,
        a line style associated with it.
    """
    return ((CMAP((x + 1) / (number + 2)), LINE_STYLES[x % len(LINE_STYLES)]) for x in range(number))


filter_func = (lambda x: x in sys.argv[3].split(",")) if len(sys.argv) == 4 else lambda x: True
