#!/usr/bin/env python3
"""A script that combines exploitabilities of different runs into one file.

Usage:
    ./scripts/exploitability_combiner.py <starting-path>
"""
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.axes as axes  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.run.save import (Output, approx_game,
                                             get_coalition_distribution)
from scripts.find_data_jsons import find_data_jsons
from scripts.plot_base import NAME_MAP, filter_func, get_colors

ALREADY_CUMULATIVE = ["best_states", "expected_best_states"]
N_COLS = 3


def add_to_plt(ax: axes.Axes, data: np.ndarray, name: str, color: Any, step: int, cumulative: bool,
               number_of_coalitions: int, minimal_game: list[Coalition], width: float, shift: float) -> tuple[Any, Any]:
    """Add data drawing to plot."""
    if not cumulative:
        labels, distribution = get_coalition_distribution(number_of_coalitions, data[step + 1],
                                                          minimal_game)
        distribution = [x for x in distribution]
    else:
        labels, distribution = get_coalition_distribution(number_of_coalitions, data[step], minimal_game)
        for i in range(step):
            distribution = [x + y for x, y in zip(distribution, get_coalition_distribution(
                number_of_coalitions, data[i], minimal_game)[1])]

    plotted = ax.bar([i + shift for i in range(len(labels))], distribution, width, color=color, zorder=4, label=name)
    return labels, plotted


def draw_combined_graph(ax: axes.Axes, chosen_coalitions: list[tuple[str, np.ndarray]],
                        output_path: Path, title: str, step: int, x_label: bool = True,
                        y_label: bool = True) -> tuple[list, list]:
    """Draw data into a combined graph."""
    colors = get_colors(len(chosen_coalitions))
    print(ax)
    ax.grid(zorder=-1, alpha=.3)
    ax.set_ylim(bottom=0, top=1)

    # compute bar with and placement
    number_of_bars = len(chosen_coalitions)
    width_of_bar = 1 / (number_of_bars + 1)
    starting_shift = -0.5 + width_of_bar

    labels: list = []
    plotted: list = []
    names: list = []
    for i, (name, coalitions) in enumerate(chosen_coalitions):
        color, _ = next(colors)
        number_of_coalitions, _, minimal_game = approx_game(coalitions)
        labels, new_plotted = add_to_plt(ax, coalitions, NAME_MAP.get(name, name), color, step,
                                         name not in ALREADY_CUMULATIVE, number_of_coalitions, minimal_game,
                                         width_of_bar, starting_shift + i * width_of_bar) or labels
        plotted.append(new_plotted)
        names.append(NAME_MAP.get(name, name))
    ax.set_xticks(range(len(labels)), labels, rotation='vertical')
    ax.tick_params(labelsize=6)
    ax.set_yticks([], [])
    ax.title.set_text(title)
    ax.title.set_fontfamily("monospace")
    ax.title.set_fontsize(8)
    # ax.autoscale()
    if x_label:
        ax.set_xlabel("Coalition", fontsize=8)
    if y_label:
        ax.set_ylabel("Probability", fontsize=8)
    # ax.savefig(output_path)
    # ax.close('all')
    return plotted, names


def main(path: Path = Path(sys.argv[1]), title: str = sys.argv[2]) -> None:
    """Run the script."""
    for data in find_data_jsons(path):
        save_path = data.parent / "cumulative_chosen_coalitions"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        with data.open("r") as f:
            data_keys = json.load(f).keys()

        # a tuple (name, chosen_coalitions) of all runs
        steps = min(len(Output.from_file(data, x).actions_list) for x in data_keys if filter_func(x))
        chosen_coalitions = [(x, Output.from_file(data, x).actions) for x in data_keys if filter_func(x)]
        fig, axs = plt.subplots(math.ceil(steps / N_COLS), N_COLS, layout='constrained')
        axs = axs.flatten()
        fig.set_size_inches(8.3, 10)

        for step in range(steps):
            step_path = save_path / f"{step + 1}.pdf"
            # steps in title are counted from 1.
            plotted, labels = draw_combined_graph(axs[step], chosen_coalitions, step_path,
                                                  f"{title} - step {step + 1}", step,
                                                  step // N_COLS + 1 == steps // N_COLS,
                                                  step % N_COLS == 0)

        fig.set_tight_layout({"pad": 1.5, "rect": [0, 0.035, 1, 1]})
        fig.legend(plotted, labels, loc='upper center', ncol=10, fontsize=8,
                   bbox_to_anchor=(0.5, 0.04))
        print(save_path.with_suffix(".pdf"))
        fig.savefig(save_path.with_suffix(".pdf"))
        # fig.close()


if __name__ == '__main__':
    main()

# fig, axs = plt.subplots(1, 2, layout='constrained')
#
# x = np.arange(0.0, 2.0, 0.02)
# y1 = np.sin(2 * np.pi * x)
# y2 = np.exp(-x)
# l1, = axs[0].plot(x, y1)
# l2, = axs[0].plot(x, y2, marker='o')
#
# y3 = np.sin(4 * np.pi * x)
# y4 = np.exp(-2 * x)
# l3, = axs[1].plot(x, y3, color='tab:green')
# l4, = axs[1].plot(x, y4, color='tab:red', marker='^')
#
# fig.legend((l1, l2), ('Line 1', 'Line 2'), loc='upper left')
# fig.legend((l3, l4), ('Line 3', 'Line 4'), loc='outside right upper')
#
# plt.show()
