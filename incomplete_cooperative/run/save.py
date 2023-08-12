"""Handle saving files output."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from argparse import Namespace

from incomplete_cooperative.coalitions import Coalition
import matplotlib.pyplot as plt  # type: ignore
import numpy as np


@dataclass
class Output:
    """Hold all the output information."""

    exploitability: np.ndarray
    actions: np.ndarray
    parsed_args: Namespace

    @property
    def avg_final_exploitability(self) -> float:
        """Compute the average of final exploitabilities."""
        return np.average(self.exploitability[-1])

    @property
    def metadata(self) -> dict:
        """Get computation metadata from parsed args."""
        args_dict = vars(self.parsed_args).copy()
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

    @property
    def json(self) -> dict:
        """Generate a dictionary representation."""
        return {"exploitability": self.exploitability_list,
                "actions": self.actions_list,
                "metadata": self.metadata}


def save_exploitability_plot(path: Path, unique_name: str, output: Output) -> None:
    """Save exploitability data to a figure."""
    if not path.exists():
        path.mkdir(parents=True)
    fig_data = output.exploitability
    data_length = len(output.exploitability_list)
    fig, ax = plt.subplots()
    plt.errorbar(
        range(data_length), np.mean(fig_data, 1), yerr=np.std(fig_data, 1))
    ax.set_ylim(bottom=0)
    plt.savefig((path / unique_name).with_suffix(".png"))


def save_draw_coalitions(path: Path, unique_name: str, output: Output) -> None:
    """Draw coalition distribution in each step."""
    unique_path = path / unique_name
    unique_path.mkdir(parents=True)
    all_data = output.actions

    plt.margins(0.2)
    for i in range(all_data.shape[0]):
        time_slice = all_data[i]
        fig, ax = plt.subplots()
        labels, counts = np.unique(time_slice, return_counts=True)
        ax.set_xticks(labels,
                      [list(Coalition(int(x)).players) for x in labels],
                      rotation='vertical')
        # transform counts to percentage
        ax.bar(labels, counts / all_data.shape[1], align='center')
        plt.autoscale()
        plt.tight_layout()
        plt.savefig((unique_path / str(i + 1)).with_suffix(".png"))


def save_json(path: Path, unique_name: str, output: Output) -> None:
    """Save the data to json."""
    data = json.loads(path.read_text()) if path.exists() else {}
    if unique_name in data.keys():
        return
    data.update({unique_name: output.json})
    with path.open("w") as f:
        json.dump(data, f, default=json_serializer)


def json_serializer(obj: Any) -> Any:
    """Serialize an object."""
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


SAVERS = {
    "exploitability_plots": save_exploitability_plot,
    "data.json": save_json,
    "chosen_coalitions": save_draw_coalitions
}


def save(model_path: Path, unique_name: str, output: Output) -> None:
    """Save the data."""
    if not model_path.exists():
        model_path.mkdir(parents=True)
    for saver_name, saver in SAVERS.items():
        saver(model_path / saver_name, unique_name, output)
