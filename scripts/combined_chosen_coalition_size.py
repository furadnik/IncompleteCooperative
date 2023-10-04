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

import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.run.save import Output, approx_game
from scripts.find_data_jsons import find_data_jsons
from scripts.plot_base import NAME_MAP, filter_func, get_colors

ALREADY_CUMULATIVE = ["best_states", "expected_best_states"]


def fixup_dist(labels: np.ndarray, distribution: np.ndarray, *, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixup the distribution to include all value sizes."""
    values = np.zeros(maximum)
    for label, value in zip(labels, distribution):
        if np.isnan(label):
            continue
        values[int(label)] = value
    return np.array(range(maximum)), values


def add_to_plt(data: np.ndarray, name: str, color: Any, step: int, cumulative: bool,
               number_of_coalitions: int, minimal_game: list[Coalition], width: float, shift: float) -> Any:
    """Add data drawing to plot."""
    if not cumulative:
        labels, distribution = fixup_dist(*np.unique(data[step + 1], return_counts=True),
                                          maximum=len(minimal_game) - 1)
    else:
        labels, distribution = fixup_dist(*np.unique(data[step], return_counts=True),
                                          maximum=len(minimal_game) - 1)
        for i in range(step):
            _, another_distribution = fixup_dist(*np.unique(data[i], return_counts=True),
                                                 maximum=len(minimal_game) - 1)
            distribution += another_distribution
    print(minimal_game)
    number_of_players = max(len(list(x.players)) for x in minimal_game) + 1
    print(number_of_players)
    coalition_counts = np.array([math.comb(number_of_players, i) for i in range(number_of_players)])
    print(coalition_counts)
    print(distribution)
    coalition_counts[0] = 1
    distribution = distribution.astype(np.float64)
    distribution /= np.sum(distribution)
    distribution *= step + 1
    distribution /= coalition_counts
    print(step, distribution, cumulative)

    plt.bar(labels + shift, distribution, width, color=color, zorder=4, label=name)
    return labels.tolist()


def draw_combined_graph(chosen_coalitions: list[tuple[str, np.ndarray]],
                        output_path: Path, title: str, step: int) -> None:
    """Draw data into a combined graph."""
    colors = get_colors(len(chosen_coalitions))
    plt.grid(zorder=-1, alpha=.3)
    plt.ylim(bottom=0, top=1)

    # compute bar with and placement
    number_of_bars = len(chosen_coalitions)
    width_of_bar = 1 / (number_of_bars + 1)
    starting_shift = -0.5 + width_of_bar

    labels: list = []
    current_maximum = 0
    for i, (name, coalitions) in enumerate(chosen_coalitions):
        color, _ = next(colors)
        number_of_coalitions, _, minimal_game = approx_game(coalitions)
        current_maximum = max(np.nanmax(coalitions), current_maximum)
        add_to_plt(coalitions, NAME_MAP.get(name, name), color, step,
                   name not in ALREADY_CUMULATIVE, number_of_coalitions, minimal_game,
                   width_of_bar, starting_shift + i * width_of_bar)
    plt.legend()
    plt.xticks(range(int(current_maximum + 1)), range(int(current_maximum + 1)))
    plt.xlim(left=2 - .6, right=current_maximum + .6)
    plt.title(title, family="monospace")
    # plt.autoscale()
    plt.tight_layout(pad=2)
    plt.xlabel("Coalition size")
    plt.ylabel("Revealed percentage")
    plt.savefig(output_path)
    plt.close('all')


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
        for step in range(steps):
            step_path = save_path / f"{step + 1}.pdf"
            # steps in title are counted from 1.
            draw_combined_graph(chosen_coalitions, step_path, f"{title} - step {step + 1}", step)


if __name__ == '__main__':
    main()
