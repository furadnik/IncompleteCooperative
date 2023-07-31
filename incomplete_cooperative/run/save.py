"""Handle saving files output."""
import json
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np


def get_metadata(parsed_args) -> dict:
    """Get metadata from parsed arguments."""
    args_dict = vars(parsed_args)
    func = args_dict.pop("func")
    args_dict["run_type"] = "eval" if "eval" in repr(func) else "learn"
    return args_dict


def save(data: np.ndarray, model_path: Path, parsed_args) -> None:
    """Save the data."""
    metadata = get_metadata(parsed_args)
    data_list = data.tolist()
    with model_path.with_suffix(".json").open("w") as f:
        json.dump({"data": data_list, "metadata": metadata}, f)
    save_fig(model_path.with_suffix(".png"), len(data_list), data)


def save_fig(fig_path: Path, data_length: int, fig_data: np.ndarray) -> None:
    """Save data to a figure."""
    fig, ax = plt.subplots()
    plt.errorbar(
        range(data_length), np.mean(fig_data, 1), yerr=np.std(fig_data, 1))
    plt.savefig(fig_path)
