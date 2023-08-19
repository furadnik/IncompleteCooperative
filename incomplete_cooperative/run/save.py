"""Handle saving files output."""
from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.protocols import Value


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

    @classmethod
    def from_file(cls, path: Path, unique_name: str) -> Output:
        """Parse the json data back to self."""
        with path.open("r") as f:
            data = json.load(f)

        return cls.from_json(data[unique_name])

    @classmethod
    def from_json(cls, data: dict) -> Output:
        """Parse the json data back to self."""
        data["metadata"]["func"] = data["metadata"]["run_type"]
        data["parsed_args"] = Namespace(**data.pop("metadata"))
        data["exploitability"] = np.array(data["exploitability"], dtype=Value)
        data["actions"] = np.array(data["actions"], dtype=int)

        return cls(**data)


def save_exploitability_plot(path: Path, unique_name: str, output: Output) -> None:
    """Save exploitability data to a figure."""
    if not path.exists():
        path.mkdir(parents=True)
    fig_data = output.exploitability
    data_length = len(output.exploitability_list)
    fig, ax = plt.subplots()
    ax.grid(zorder=-1)
    mean = np.mean(fig_data, 1)
    stde = np.std(fig_data, 1)
    plt.plot(
        range(data_length), mean, zorder=3)
    plt.fill_between(
        range(data_length), mean + stde, mean - stde, zorder=3, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.savefig((path / unique_name).with_suffix(".png"))
    plt.close('all')


def save_draw_coalitions(path: Path, unique_name: str, output: Output) -> None:
    """Draw coalition distribution in each step."""
    unique_path = path / unique_name
    unique_path.mkdir(parents=True)
    all_data = output.actions
    number_of_coalitions = int(np.max(all_data)) + 1

    plt.margins(0.2)
    for i in range(all_data.shape[0]):
        time_slice = all_data[i]
        fig, ax = plt.subplots()
        labels, some_counts = np.unique(time_slice, return_counts=True)
        counts = np.zeros(number_of_coalitions)

        for label, count in zip(labels, some_counts / all_data.shape[1]):
            counts[int(label)] = count

        # combine them together
        labels_with_counts = sorted(zip(
            # generate coalitions
            (list(Coalition(int(x)).players) for x in range(number_of_coalitions)), counts),
            key=lambda x: len(x[0]))  # sort by coalition

        # now split them apart again, but sorted
        new_labels = [x[0] for x in labels_with_counts]
        new_counts = [x[1] for x in labels_with_counts]

        ax.grid(zorder=-1)
        ax.set_xticks(range(number_of_coalitions),
                      new_labels,
                      rotation='vertical')
        ax.bar(range(number_of_coalitions), new_counts, align='center', zorder=3)
        plt.autoscale()
        plt.tight_layout()
        plt.savefig((unique_path / str(i + 1)).with_suffix(".png"))
        plt.close('all')


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
