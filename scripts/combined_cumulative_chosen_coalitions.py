#!/usr/bin/env python3
"""A script that combines exploitabilities of different runs into one file.

Usage:
    ./scripts/exploitability_combiner.py <starting-path>
"""
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.run.save import (Output, approx_game,
                                             get_coalition_distribution)
from scripts.find_data_jsons import find_data_jsons
from scripts.plot_base import NAME_MAP, get_colors

ALREADY_CUMULATIVE = ["best_states"]


def add_to_plt(data: np.ndarray, name: str, color: Any, step: int, cumulative: bool,
               number_of_coalitions: int, minimal_game: list[Coalition], width: float, shift: float) -> Any:
    """Add data drawing to plot."""
    if not cumulative:
        labels, distribution = get_coalition_distribution(number_of_coalitions, data[step], minimal_game)
    else:
        labels, distribution = get_coalition_distribution(number_of_coalitions, data[step], minimal_game)
        for i in range(step):
            distribution = [x + y for x, y in zip(distribution, get_coalition_distribution(
                number_of_coalitions, data[i], minimal_game)[1])]

    plt.bar([i + shift for i in range(len(labels))], distribution, width, color=color, zorder=4, label=name)
    return labels


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
    for i, (name, coalitions) in enumerate(chosen_coalitions):
        color = next(colors)
        number_of_coalitions, _, minimal_game = approx_game(coalitions)
        labels = add_to_plt(coalitions, NAME_MAP.get(name, name), color, step,
                            name not in ALREADY_CUMULATIVE, number_of_coalitions, minimal_game,
                            width_of_bar, starting_shift + i * width_of_bar) or labels
    plt.legend()
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.title(title, family="monospace")
    plt.autoscale()
    plt.tight_layout(pad=2)
    plt.xlabel("Coalition")
    plt.ylabel("Probability")
    plt.savefig(output_path)
    plt.close('all')


def main(path: Path = Path(sys.argv[1]), title: str = sys.argv[2]) -> None:
    """Run the script."""
    for data in find_data_jsons(path):
        save_path = data.parent / "cumulative_chosen_coalitions"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        with data.open("r") as f:
            data_keys = json.load(f).keys()

        # a tuple (name, chosen_coalitions) of all runs
        steps = min(len(Output.from_file(data, x).actions_list) for x in data_keys)
        chosen_coalitions = [(x, Output.from_file(data, x).actions) for x in data_keys]
        for step in range(steps):
            step_path = save_path / str(step + 1)
            # steps in title are counted from 1.
            draw_combined_graph(chosen_coalitions, step_path, f"{title} - step {step + 1}", step)


if __name__ == '__main__':
    main()
