"""Import the example from SHAP (https://github.com/shap/shap/blob/master/notebooks/api_examples/explainers/Exact.ipynb)."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import shap
import xgboost

from incomplete_cooperative.coalitions import all_coalitions
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.game_properties import is_superadditive

parser = ArgumentParser(description="Turn a basic SHAP example into a dataset of subadditive set functions.")
parser.add_argument("--filepath", type=Path, default=Path("shap_example_set_functions.npy"))


def main(args=parser.parse_args()) -> None:
    """Turn a basic SHAP example into a dataset of subadditive set functions."""
    # get a dataset on income prediction
    filepath = args.filepath

    X, y = shap.datasets.adult()
    X = X.values

    number_of_features = X.shape[1]
    number_of_samples = X.shape[0]

    set_functions = np.zeros((number_of_samples, 2 ** number_of_features))

    for i, coalition in enumerate(all_coalitions(number_of_features)):
        if not coalition:
            continue
        # train an XGBoost model (but any other model type would also work)
        model = xgboost.XGBClassifier()
        model.fit(X[:, list(coalition.players)], y)
        predictions = model.predict_proba(X[:, list(coalition.players)])[:, 1]
        set_functions[:, i] = predictions
        if i % 100 == 0:
            print(f"Processed {i} coalitions")

    subadditive_count = 0
    subadditive = []
    for i in range(number_of_samples):
        game = IncompleteCooperativeGame(number_of_features)
        game.set_values(-set_functions[i])
        if is_superadditive(game):
            subadditive_count += 1
            subadditive.append(i)
        print("Sample", i + 1, "processed")
        print(f"Subadditive games: {subadditive_count / (i + 1) * 100:.2f}%")

    np.save(filepath, -set_functions[subadditive])
    print(f"Saved {len(subadditive)} subadditive set functions to {filepath}")
