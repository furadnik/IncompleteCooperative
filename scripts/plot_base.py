"""Base stuff regarding plotting."""
import os
import sys

import matplotlib as mpl  # type: ignore
from matplotlib import colormaps

COALITION = os.environ.get("ICG_COALITION", "Subset")
PLAYER = os.environ.get("ICG_PLAYER", "Element")
EXPLOITABILITY = os.environ.get("ICG_EXPLOITABILITY", "$l_1$ Divergence")
FORMAT = os.environ.get("ICG_FORMAT", "aaai")

if FORMAT == "aamas":
    text = {'text.latex.preamble': '\\usepackage{libertine}\n\\renewcommand{\\ttdefault}{cmtt}', }
elif FORMAT == "aaai":
    text = {'font.family': "times"}
else:
    text = {'font.family': "serif"}


mpl.rcParams.update({
    'axes.labelsize': 16,
    'font.size': 16,
    **text,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': [7.2, 4.5],
    'text.usetex': True
})

CMAP = colormaps["inferno"]
NAME_MAP = {k: "{\\sc " + v + "}" for k, v in {
    "random_eval": "Random",
    "eval": "PPO",
    "eval_ln": "PPO+LN",
    "eval_lin": "PPO+Lin",
    "eval_ln_lin": "PPO+LN+Lin",
    "solve_greedy": "Oracle Greedy",
    "solve_greedy_worst": "Oracle Greedy - Worst",
    "solve_ugreedy": "Oracle Uniform Greedy",
    "best_states": "Oracle Optimal",
    "expected_greedy": "Offline Greedy",
    "expected_ugreedy": "Offline Uniform Greedy",
    "expected_best_states": "Offline Optimal",
    "largest_coalitions": f"Largest {COALITION}s",
}.items()}
LINE_STYLES = ["solid", "dashed", "dashdot", "dotted"]

MULTIFIG_SIZE = (8.3, 9.1) if FORMAT != "bakalarka" else (6, 9)
MULTIFIG_PAD = .45 if FORMAT != "bakalarka" else .81
MULTIFIG_RECT = [0, 0.035, 1, 1] if FORMAT != "bakalarka" else [0, 0.08, 1, 1]
PLOT_RATIO = 1.2 / 2


COLOR_VALUES = {k: (CMAP(i / (len(NAME_MAP.keys()))), LINE_STYLES[i % len(LINE_STYLES)])
                for i, k in enumerate(sorted(NAME_MAP.keys(), key=NAME_MAP.get))}  # type: ignore

TICK_SIZE = 9
LABEL_SIZE = 10
TITLE_SIZE = 11

filter_func = (lambda x: x in sys.argv[3].split(",")) if len(sys.argv) >= 4 else lambda x: True

ALREADY_CUMULATIVE = ["best_states", "expected_best_states"]
N_COLS = 3
COMBINED_COAL_MAX_STEPS = int(os.environ.get("COMBINED_COAL_MAX_STEPS", 12))
if FORMAT == "bakalarka":
    LEGEND_N_COLS = 3
else:
    LEGEND_N_COLS = 10
