#!/usr/bin/env python3
"""A script that combines exploitabilities of different runs into one file.

Usage:
    ./scripts/exploitability_combiner.py <starting-path>
"""
import json
import sys
from itertools import product, zip_longest
from pathlib import Path
from typing import Callable

import numpy as np
import scipy as sp

from incomplete_cooperative.coalitions import Coalition, all_coalitions
from incomplete_cooperative.run.save import Output
from scripts.find_data_jsons import find_data_jsons


def actions_to_chosen(actions: np.ndarray, time: int, number_of_coalitions: int) -> np.ndarray:
    """Turn the chosen actions from Output into an array of [time, n_coalitions], with percentage of being chosen at a given timestep."""
    revealed = np.zeros((time, number_of_coalitions))
    if len(actions) == time + 1:
        actions = actions[1:]
    already_cumulative = len(actions.shape) == 3
    for step, action in enumerate(actions):
        action = action[np.logical_not(np.logical_or(np.isnan(action), action == 0))].astype(int)
        revealed[step] += np.eye(number_of_coalitions)[action].sum(axis=0) / (len(action) or 1)
        if not already_cumulative and step > 0:
            revealed[step] += revealed[step - 1]
        elif already_cumulative:
            revealed[step] *= step + 1
    return revealed


def print_out_by_size(revealed: np.ndarray, number_of_players: int, names: list[str]) -> None:
    """Print out revealed coalition percentage."""
    coalition_lengths = [len(coal) for coal in all_coalitions(number_of_players)]
    viable_lengths = np.arange(2, number_of_players)
    coalition_length_count = sp.special.comb(number_of_players, viable_lengths)

    revealed_counts = np.array([revealed[:, :, coalition_lengths == length].sum(axis=-1) / count
                                for length, count in zip(viable_lengths, coalition_length_count)
                                ])
    revealed_counts = np.moveaxis(revealed_counts, 0, -1)
    assert revealed_counts.shape[:-1] == revealed.shape[:-1]

    _print_out(revealed_counts, names, np.arange(number_of_players - 2), lambda x: str(x + 2))


def format_coalition(coal: int) -> str:
    """Format coalition to string."""
    return f"[{",".join(str(p + 1) for p in Coalition(coal).players)}]"


def print_out(revealed: np.ndarray, number_of_players: int, names: list[str]) -> None:
    """Print out revealed coalition percentage."""
    viable_coalition_ids = np.array([coal.id for coal in sorted(all_coalitions(number_of_players), key=len)
                                     if len(coal) not in [0, 1, number_of_players]])
    print(*(f"{{{format_coalition(coal)}}}" for coal in viable_coalition_ids), sep=",", file=sys.stderr)
    _print_out(revealed, names, viable_coalition_ids,
               format_coalition)


def _print_out(revealed: np.ndarray, names: list[str], viable: np.ndarray, label_map: Callable) -> None:
    """Do the actual printing of only the `viable` indices."""
    n_approaches, n_steps, _ = revealed.shape
    assert n_approaches == len(names)

    row_ids = list(product(range(n_approaches), range(n_steps)))
    print("coalition", *(f"{names[approach]}_{step + 1}" for approach, step in row_ids))
    for item_id in viable:
        print(label_map(item_id), *(revealed[approach, step, item_id] for approach, step in row_ids))


def main(path: Path = Path(sys.argv[1]), group_by_size: bool = "--group-by-size" in sys.argv) -> None:
    """Run the script."""
    for data in find_data_jsons(path):
        with data.open("r") as f:
            data_keys = json.load(f).keys()

        n_approaches = len(data_keys)
        steps = min(len(Output.from_file(data, x).actions_list) for x in data_keys)
        # a tuple (name, chosen_coalitions) of all runs
        chosen_coalitions = [(x, Output.from_file(data, x).actions) for x in data_keys]

        # get number of players
        numbers_of_players = (Output.from_file(data, x).parsed_args.number_of_players for x in data_keys)
        number_of_players = next(numbers_of_players)
        assert all(num == number_of_players for num in numbers_of_players)

        number_of_coalitions = 2**number_of_players
        revealed = np.zeros((n_approaches, steps, number_of_coalitions))

        names = []
        for approach, (name, actions) in enumerate(chosen_coalitions):
            names.append(name)
            revealed[approach] = actions_to_chosen(actions, steps, number_of_coalitions)

        if group_by_size:
            print_out_by_size(revealed, number_of_players, names)
        else:
            print_out(revealed, number_of_players, names)


if __name__ == '__main__':
    main()
