"""This is a script for comparing the algorithms for multiplicative estimation with our approach."""
import argparse
import signal
import sys
from os import getenv
from pathlib import Path

import numpy as np
import scipy

from incomplete_cooperative.bounds import BOUNDS
from incomplete_cooperative.coalitions import (Coalition,
                                               minimal_game_coalitions)
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.generators import GENERATORS
from incomplete_cooperative.multiplicative import APPROXIMATORS
from incomplete_cooperative.multiplicative.multiplicative_factor import (
    mul_factor_lower_upper_bound, mul_factor_upper_to_approximation)
from incomplete_cooperative.run.model import GAP_FUNCTIONS
from incomplete_cooperative.run.save import Output


def get_action_sequence(model_path: Path, name: str) -> list[Coalition]:
    """Get the action sequence from the model."""
    output = Output.from_file(model_path, name)
    return [Coalition(int(action)) for action in output.actions[:, 0]]


def run(out_data: list[tuple[float, ...]], args: argparse.Namespace) -> None:
    """Compare multiplicative algos to ours."""
    game_generator = GENERATORS[args.game_generator]
    bounds_computer = BOUNDS[args.bounds_computer]
    approximator = APPROXIMATORS[args.approximator]
    divergence = GAP_FUNCTIONS[args.divergence]
    samples = args.samples
    rng = np.random.default_rng(args.seed)
    number_of_players = args.number_of_players
    our_action_sequence = get_action_sequence(args.model, args.model_name)

    incomplete_game = IncompleteCooperativeGame(number_of_players, bounds_computer)
    minimal_game = list(minimal_game_coalitions(number_of_players))
    print(minimal_game)

    print("sample", "len_queried", "approx_to_upper", "lower_to_upper_theirs", "lower_to_upper_ours", "divg_theirs", "divg_ours")
    sample = 0
    while samples is None or sample < samples:
        game = game_generator(number_of_players, rng)

        # their approximation
        queried_coal_ids, approximated_game = approximator(-game)
        approximated_game = -approximated_game
        pure_queried_coals = set(map(Coalition, map(int, queried_coal_ids))).difference(minimal_game)
        queried_coals = list(pure_queried_coals.union(minimal_game))
        budget = len(pure_queried_coals)  # only count those which weren't in the minimal game

        # compute our ratio
        revealed_sequence = our_action_sequence[:budget]
        incomplete_game.set_known_values(game.get_values(minimal_game + revealed_sequence),
                                         minimal_game + revealed_sequence)
        incomplete_game.compute_bounds()
        lower_to_upper_ours = mul_factor_lower_upper_bound(-incomplete_game)
        divg_ours = divergence(incomplete_game)

        # set up incomplete game from their revealed
        incomplete_game.set_known_values(game.get_values(queried_coals), queried_coals)
        incomplete_game.compute_bounds()

        # compute their ratios
        upper_to_approx = mul_factor_upper_to_approximation(-approximated_game, -incomplete_game)
        lower_to_upper_theirs = mul_factor_lower_upper_bound(-incomplete_game)
        divg_theirs = divergence(incomplete_game)

        sample += 1
        out_data.append((budget, upper_to_approx, lower_to_upper_theirs, lower_to_upper_ours, divg_theirs, divg_ours))
        # print(*out_data[-1])
    _callback_fn()


out_data: list[tuple[float, ...]] = []


def _callback_fn(a=None, b=None) -> None:
    data = np.array(out_data)
    samples = len(data)
    print(samples, *data.mean(axis=0))
    print(samples, *data.std(axis=0))
    print(samples, *scipy.stats.sem(data, axis=0))  # type: ignore[misc]
    sys.exit(0)


def main() -> None:
    """Entrypoint for script."""
    parser = argparse.ArgumentParser(description='Compare multiplicative algorithms.')
    parser.add_argument('samples', type=int, nargs='?', default=None)
    parser.add_argument('--game_generator', type=str, default="k_budget_generator")
    parser.add_argument('--number_of_players', type=int, default=6)
    parser.add_argument('--bounds_computer', type=str, default="sam_apx_1")
    parser.add_argument('--approximator', type=str, default="max_xos")
    parser.add_argument('--divergence', type=str, choices=GAP_FUNCTIONS.keys(), default="l1_norm")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--model_name', type=str, default="expected_greedy")

    signal.signal(signal.SIGINT, _callback_fn)
    run(out_data, parser.parse_args())


if __name__ == '__main__':
    main()
