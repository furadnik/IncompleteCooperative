#!/usr/bin/env python3
"""A script that combines exploitabilities of different runs into one file.

Usage:
    ./scripts/exploitability_combiner.py <starting-path>
"""
import json
import math
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Any

import matplotlib.axes as axes  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.run.save import (Output, approx_game,
                                             get_coalition_distribution2)
from scripts.find_data_jsons import find_data_jsons
from scripts.plot_base import (ALREADY_CUMULATIVE, COALITION, COLOR_VALUES,
                               COMBINED_COAL_MAX_STEPS, LABEL_SIZE,
                               LEGEND_N_COLS, MULTIFIG_PAD, MULTIFIG_RECT,
                               MULTIFIG_SIZE, N_COLS, NAME_MAP, TITLE_SIZE,
                               filter_func)

TICK_SIZE = 3


def get_labels_and_distribution_for_step(step: int, data: np.ndarray, cumulative: bool
                                         ) -> tuple[list[list[int]], list[float]]:
    """Add data drawing to plot."""
    if not cumulative:
        labels, distribution = get_coalition_distribution2(data[step + 1])
        distribution = [x for x in distribution]
    else:
        labels, distribution = get_coalition_distribution2(data[step])
        for i in range(step):
            distribution = [x + y for x, y in zip(distribution, get_coalition_distribution2(data[i])[1])]
    assert [len(x) for x in labels], f"{step}, {data}, {cumulative}"
    return [list(x) for x in labels], distribution


def draw_combined_graph(chosen_coalitions: list[tuple[str, np.ndarray]], step: int) -> tuple[list[list[int]], list[str], list[list[float]]]:
    """Transform the raw chosen coalitions into distributions and labels; for step `step`.

    Return the labels (names of coalitions), and the names (of algorithms) with step number, and the coalition distributions.
    """
    labels: list = []
    dists: list = []
    names: list = []
    for name, coalitions in chosen_coalitions:
        labels, dist_for_name = get_labels_and_distribution_for_step(step, coalitions, name not in ALREADY_CUMULATIVE)
        names.append(f"{name}_{step}")
        dists.append(dist_for_name)

    return labels, names, dists


def group_coalitions_by_size(full_labels: list[list[int]], full_dists: list[float]) -> tuple[list[int], list[float]]:
    """Group coalition distributions by their size."""
    max_len = max(len(x) for x in full_labels)
    labels = [i + 1 for i in range(max_len)]
    dists = []
    for label in labels:
        dists_of_size = [full_dists[i] for i in range(len(full_dists)) if len(full_labels[i]) == label]
        if dists_of_size:
            dists.append(sum(dists_of_size) / len(dists_of_size))
        else:
            dists.append(0)
    return labels, dists


def main(path: Path = Path(sys.argv[1]), group_by_size: bool = "--group-by-size" in sys.argv) -> None:
    """Run the script."""
    for data in find_data_jsons(path):
        with data.open("r") as f:
            data_keys = json.load(f).keys()

        steps = min(len(Output.from_file(data, x).actions_list) for x in data_keys)
        # a tuple (name, chosen_coalitions) of all runs
        chosen_coalitions = [(x, Output.from_file(data, x).actions) for x in data_keys]

        labels = None
        dists = []
        names = []
        for step in range(steps):
            step_labels, step_names, step_dists = draw_combined_graph(chosen_coalitions, step)

            if group_by_size:
                for i in range(len(step_dists)):
                    print(step_names[i], step_labels, file=sys.stderr)
                    new_step_labels, step_dists[i] = group_coalitions_by_size(step_labels, step_dists[i])
                step_labels = new_step_labels  # type: ignore[assignment]

            if labels is None:
                labels = step_labels
            else:
                assert labels == step_labels

            dists += step_dists
            names += step_names

        assert labels is not None
        print("coalition", *names)
        print(len(labels), len(dists), file=sys.stderr)
        print(*[len(x) for x in dists], file=sys.stderr)
        str_labels = [str(x).replace(' ', '') for x in labels]
        for line in zip_longest(str_labels, *dists, fillvalue=0):
            print(*line, sep='\t')


if __name__ == '__main__':
    main()
