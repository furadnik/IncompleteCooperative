"""Validate that best is actually always best."""
from pathlib import Path
from sys import argv

import numpy as np

from incomplete_cooperative.run.save import Output

from .find_data_jsons import find_data

EPSILON = 0.000000001


def validate_data(data: dict[str, Output]) -> None:
    """Return output."""
    if "best_states" not in data.keys():
        print("No best, nothing to validate.")
        return
    # make them a little better to account for float precision
    best_exploitabilities = data["best_states"].avg_exploitabilities - EPSILON

    for key, output in data.items():
        exploitability = output.exploitability
        for i in range(exploitability.shape[0]):
            assert np.all(exploitability[i] >= best_exploitabilities[i]), \
                f"{key}, ex:{exploitability[i].tolist()}, bst:{best_exploitabilities[i]}"
    print("OK.")


def main(path=Path(argv[1])) -> None:
    """Validate in a path."""
    for data in find_data(path):
        validate_data(data)
