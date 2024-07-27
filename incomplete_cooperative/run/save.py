"""Handle saving files output."""
from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from incomplete_cooperative.coalitions import (Coalition,
                                               minimal_game_coalitions)
from incomplete_cooperative.protocols import Player, Value
from incomplete_cooperative.run.model import ModelInstance


@dataclass
class Output:
    """Hold all the output information."""

    data: np.ndarray
    actions: np.ndarray
    parsed_args: Namespace

    @property
    def data_avg_final(self) -> float:
        """Compute the average of final exploitabilities."""
        return np.average(self.data[-1])

    @property
    def avg_data(self) -> np.ndarray:
        """Compute the average of exploitabilities."""
        return np.mean(self.data, 1)

    @property
    def metadata(self) -> dict:
        """Get computation metadata from parsed args."""
        args_dict = vars(self.parsed_args).copy()
        func = args_dict.pop("func")
        args_dict["run_type"] = "eval" if "eval" in repr(func) else "learn"
        return args_dict

    @property
    def data_list(self) -> list[list[float]]:
        """Turn data to a list."""
        return self.data.tolist()

    @property
    def actions_list(self) -> list[list[float]]:
        """Turn data to a list."""
        return self.actions.tolist()

    @property
    def json(self) -> dict:
        """Generate a dictionary representation."""
        return {"data": self.data_list,
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
        data["data"] = np.array(data["data"], dtype=Value)
        data["actions"] = np.array(data["actions"])

        return cls(**data)

    @property
    def instance(self) -> ModelInstance:  # pragma: no cover
        """Get another instance from the parsed args."""
        return ModelInstance.from_parsed_arguments(self.parsed_args)


def get_outputs_from_file(path: Path) -> dict[str, Output]:
    """Get outputs from `data.json` path."""
    with path.open("r") as f:
        data = json.load(f)
    return get_outputs(data)


def get_outputs(data: dict[str, Any]) -> dict[str, Output]:
    """Get outputs from `data`."""
    return {key: Output.from_json(value) for key, value in data.items()}


def save_data_plot(path: Path, unique_name: str, output: Output) -> None:
    """Save data to a figure."""
    if not path.exists():
        path.mkdir(parents=True)
    fig_data = output.data
    data_length = len(output.data_list)
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


def get_coalition_distribution(number_of_coalitions: int, data: np.ndarray,
                               minimal_game: list[Coalition]) -> tuple[list[list[Player]], list[float]]:
    """Get the labels (players) and distribution of coalitions chosen at a given time.

    Arguments:
        number_of_coalitions: the total coalitions in the game (lower estimate).
        data: the chosen coalitions in a given time slice.
        minimal_game: the coalitions in a minimal game (will not be shown in the plot).
    """
    labels, some_counts = np.unique(data, return_counts=True)
    counts = np.zeros(number_of_coalitions)

    for label, count in zip(labels, some_counts / data.shape[0]):
        if not np.isnan(label):
            counts[int(label)] = count

    # combine them together
    all_coals_with_counts = zip((Coalition(x) for x in range(number_of_coalitions)), counts)

    # filter out minimal game's coalitions, get just coal's players, sort by coal length
    labels_with_counts = sorted(
        ((list(coal.players), n) for coal, n in all_coals_with_counts if coal not in minimal_game),
        key=lambda x: len(x[0])
    )

    # now split them apart again, but sorted
    new_labels = [x[0] for x in labels_with_counts]
    new_counts = [x[1] for x in labels_with_counts]
    return new_labels, new_counts


def approx_game(all_data: np.ndarray) -> tuple[int, int, list[Coalition]]:
    """Return approximations about the game.

    Returns a tuple:
        The number of coalitions (lower bound).
        The number of players (lower bound).
        The coalitions of minimal game according to the approx number of players.
    """
    # approximate the number of coalitions: at least as many as the biggest id of a chosen one.
    number_of_coalitions = int(np.max(all_data, initial=-1,
                                      where=(np.logical_not(np.isnan(all_data)))  # exclude nans
                                      )) + 1
    approx_number_of_players = max(Coalition(number_of_coalitions).players) + 1
    minimal_game = list(minimal_game_coalitions(approx_number_of_players))
    return number_of_coalitions, approx_number_of_players, minimal_game


def save_draw_coalitions(path: Path, unique_name: str, output: Output) -> None:
    """Draw coalition distribution in each step."""
    unique_path = path / unique_name
    unique_path.mkdir(parents=True)
    all_data = output.actions
    number_of_coalitions, approx_number_of_players, minimal_game = approx_game(all_data)

    plt.margins(0.2)
    for i in range(all_data.shape[0]):
        time_slice = all_data[i]
        fig, ax = plt.subplots()
        labels, distribution = get_coalition_distribution(number_of_coalitions, time_slice,
                                                          minimal_game)

        ax.set_ylim(bottom=0, top=1)
        ax.grid(zorder=-1)
        ax.set_xticks(range(len(labels)),
                      [str(x) for x in labels],
                      rotation='vertical')
        ax.bar(range(len(labels)), distribution, align='center', zorder=3)
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
    "data_plots": save_data_plot,
    "data.json": save_json,
    "chosen_coalitions": save_draw_coalitions
}


def save(model_path: Path, unique_name: str, output: Output) -> None:
    """Save the data."""
    if not model_path.exists():
        model_path.mkdir(parents=True)
    for saver_name, saver in SAVERS.items():
        saver(model_path / saver_name, unique_name, output)
