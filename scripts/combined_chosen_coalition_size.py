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
from incomplete_cooperative.run.save import Output, approx_game
from scripts.find_data_jsons import find_data_jsons
from scripts.plot_base import (MULTIFIG_RECT, MULTIFIG_SIZES, NAME_MAP,
                               filter_func, get_colors)

ALREADY_CUMULATIVE = ["best_states", "expected_best_states"]
N_COLS = 3


def fixup_dist(labels: np.ndarray, distribution: np.ndarray, *, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixup the distribution to include all value sizes."""
    values = np.zeros(maximum)
    for label, value in zip(labels, distribution):
        if np.isnan(label):
            continue
        values[int(label)] = value
    return np.array(range(maximum)), values


def add_to_plt(ax: axes.Axes, data: np.ndarray, name: str, color: Any, step: int, cumulative: bool,
               number_of_coalitions: int, minimal_game: list[Coalition], width: float, shift: float) -> Any:
    """Add data drawing to plot."""
    number_of_players = max(len(list(x.players)) for x in minimal_game) + 2
    print(number_of_players)
    if not cumulative:
        labels, distribution = fixup_dist(*np.unique(data[step + 1], return_counts=True),
                                          maximum=number_of_players)
    else:
        labels, distribution = fixup_dist(*np.unique(data[step], return_counts=True),
                                          maximum=number_of_players)
        for i in range(step):
            _, another_distribution = fixup_dist(*np.unique(data[i], return_counts=True),
                                                 maximum=number_of_players)
            distribution += another_distribution
    print(minimal_game)
    coalition_counts = np.array([math.comb(number_of_players, i) for i in range(number_of_players)])
    print(coalition_counts)
    print(distribution)
    coalition_counts[0] = 1
    distribution = distribution.astype(np.float64)
    distribution /= np.sum(distribution)
    distribution *= step + 1
    distribution /= coalition_counts
    print(step, distribution, cumulative)

    plotted = ax.bar(labels + shift, distribution, width, color=color, zorder=4, label=name)
    return labels.tolist(), plotted


def draw_combined_graph(ax: axes.Axes, chosen_coalitions: list[tuple[str, np.ndarray]],
                        output_path: Path, title: str, step: int, x_label: bool = True,
                        y_label: bool = True) -> tuple[list, list]:
    """Draw data into a combined graph."""
    colors = get_colors(len(chosen_coalitions))
    ax.grid(zorder=-1, alpha=.3)
    ax.set_ylim(bottom=0, top=1)

    # compute bar with and placement
    number_of_bars = len(chosen_coalitions)
    width_of_bar = 1 / (number_of_bars + 1)
    starting_shift = -0.5 + width_of_bar

    current_maximum = 0
    plotted: list = []
    names: list = []
    for i, (name, coalitions) in enumerate(chosen_coalitions):
        color, _ = next(colors)
        number_of_coalitions, _, minimal_game = approx_game(coalitions)
        current_maximum = max(np.nanmax(coalitions), current_maximum)
        _, new_plotted = add_to_plt(ax, coalitions, NAME_MAP.get(name, name), color, step,
                                    name not in ALREADY_CUMULATIVE, number_of_coalitions, minimal_game,
                                    width_of_bar, starting_shift + i * width_of_bar)
        plotted.append(new_plotted)
        names.append(NAME_MAP.get(name, name))

    ax.set_xticks(range(int(current_maximum + 1)), range(int(current_maximum + 1)))
    ax.set_xlim(left=2 - .6, right=current_maximum + .6)
    ax.tick_params(labelsize=6)
    ax.title.set_text(title)
    ax.title.set_fontfamily("monospace")
    ax.title.set_fontsize(8)
    if x_label:
        ax.set_xlabel("Coalition Size", fontsize=8)

    indices = [x * 0.2 for x in range(6)]
    if y_label:
        ax.set_ylabel("Revealed Percentage", fontsize=8)
        ax.set_yticks(indices)
    else:
        ax.set_yticks(indices, [""] * len(indices))
    return plotted, names


def main(path: Path = Path(sys.argv[1]), title: str = sys.argv[2]) -> None:
    """Run the script."""
    for data in find_data_jsons(path):
        save_path = data.parent / "cumulative_chosen_coalition_sizes"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        with data.open("r") as f:
            data_keys = json.load(f).keys()

        def _coalition_to_size(coal: float) -> float:
            """Get a size of a coalition by id."""
            return np.nan if np.all(np.isnan([coal])) else len(Coalition(int(coal)))

        vcoal_to_size = np.vectorize(_coalition_to_size)

        # a tuple (name, chosen_coalitions) of all runs
        steps = min(len(Output.from_file(data, x).actions_list) for x in data_keys if filter_func(x))
        chosen_coalitions = [(x, vcoal_to_size(Output.from_file(data, x).actions)) for x in data_keys if filter_func(x)]
        # a tuple (name, chosen_coalitions) of all runs
        fig, axs = plt.subplots(math.ceil(steps / N_COLS), N_COLS, layout='constrained')
        axs = axs.flatten()
        fig.set_size_inches(MULTIFIG_SIZES)

        for step in range(steps):
            step_path = save_path / f"{step + 1}.pdf"
            # steps in title are counted from 1.
            plotted, labels = draw_combined_graph(axs[step], chosen_coalitions, step_path,
                                                  f"{title} - step {step + 1}", step,
                                                  step // N_COLS + 1 == steps // N_COLS,
                                                  step % N_COLS == 0)

        fig.set_tight_layout({"pad": 1, "rect": MULTIFIG_RECT})
        fig.legend(plotted, labels, loc='upper center', ncol=10, fontsize=8,
                   bbox_to_anchor=(0.5, 0.04))
        print(save_path.with_suffix(".pdf"))
        fig.savefig(save_path.with_suffix(".pdf"))
        # fig.close()


if __name__ == '__main__':
    main()
