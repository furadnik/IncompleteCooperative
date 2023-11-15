"""Base stuff regarding plotting."""
import sys

import matplotlib as mpl  # type: ignore
from matplotlib import colormaps

mpl.rcParams.update({
    'axes.labelsize': 16,
    'font.size': 16,
    'text.latex.preamble': '\\usepackage{libertine}\n\\renewcommand{\\ttdefault}{cmtt}',
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': [6.5, 5.5],
    'text.usetex': True
})

CMAP = colormaps["inferno"]
NAME_MAP = {k: "{\\sc " + v + "}" for k, v in {
    "random_eval": "Random",
    "eval": "PPO",
    "solve_greedy": "Oracle Greedy",
    "solve_ugreedy": "Oracle Uniform Greedy",
    "best_states": "Oracle Optimal",
    "expected_greedy": "Offline Greedy",
    "expected_ugreedy": "Offline Uniform Greedy",
    "expected_best_states": "Offline Optimal",
    "largest_coalitions": "Largest Coalitions",
}.items()}
LINE_STYLES = ["solid", "dashed", "dashdot", "dotted"]

MULTIFIG_SIZES = (8.3, 9)
MULTIFIG_RECT = [0, 0.035, 1, 1]

COLOR_VALUES = {k: (CMAP(i / (len(NAME_MAP.keys()))), LINE_STYLES[i % len(LINE_STYLES)])
                for i, k in enumerate(sorted(NAME_MAP.keys(), key=NAME_MAP.get))}  # type: ignore

TICK_SIZE = 9
LABEL_SIZE = 10
TITLE_SIZE = 11

filter_func = (lambda x: x in sys.argv[3].split(",")) if len(sys.argv) == 4 else lambda x: True
