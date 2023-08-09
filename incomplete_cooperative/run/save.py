"""Handle saving files output."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import numpy as np


@dataclass
class Output:
    """Hold all the output information."""

    exploitability: np.ndarray
    actions: np.ndarray
    parsed_args: Any

    @property
    def metadata(self) -> dict:
        """Get computation metadata from parsed args."""
        args_dict = vars(self.parsed_args)
        func = args_dict.pop("func")
        args_dict["run_type"] = "eval" if "eval" in repr(func) else "learn"
        return args_dict

    @property
    def exploitability_list(self) -> list[list[float]]:
        """Turn exploitability to a list."""
        return self.exploitability.tolist()

    @property
    def actions_list(self) -> list[list[float]]:
        """Turn exploitability to a list."""
        return self.actions.tolist()


def save_exploitability(path: Path, output: Output) -> None:
    """Save exploitability data to a figure."""
    fig_data = output.exploitability
    data_length = len(output.exploitability_list)
    fig, ax = plt.subplots()
    plt.errorbar(
        range(data_length), np.mean(fig_data, 1), yerr=np.std(fig_data, 1))
    plt.savefig(path)


def save_json(path: Path, output: Output) -> None:
    """Save the data to json."""
    with path.open("w") as f:
        json.dump({"data": output.exploitability_list,
                   "actions": output.actions_list,
                   "metadata": output.metadata}, f,
                  default=json_serializer)


def json_serializer(obj: Any) -> Any:
    """Serialize an object."""
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


SAVERS = {
    "exploitability.png": save_exploitability,
    "data.json": save_json
}


def save(model_path: Path, output: Output) -> None:
    """Save the data."""
    if not model_path.exists():
        model_path.mkdir(parents=True)
    for name, saver in SAVERS.items():
        saver(model_path / name, output)
