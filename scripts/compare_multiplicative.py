"""This is a script for comparing the algorithms for multiplicative estimation with our approach."""
from os import getenv

import numpy as np

from incomplete_cooperative.bounds import BOUNDS
from incomplete_cooperative.coalitions import (Coalition,
                                               minimal_game_coalitions,
                                               player_to_coalition)
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.game_properties import is_superadditive
from incomplete_cooperative.generators import GENERATORS
from incomplete_cooperative.multiplicative import APPROXIMATORS
from incomplete_cooperative.multiplicative.multiplicative_factor import (
    mul_factor_lower_upper_bound, mul_factor_upper_to_approximation)

OUR_ACTION_SEQUENCE: list[Coalition] = [Coalition(int(x))
                                        for x in (getenv("ACTION_SEQUENCE") or "").split(",")
                                        ] if getenv("ACTION_SEQUENCE") else []
GAME_GENERATOR = GENERATORS[getenv("GAME_CLASS") or "k_budget_generator"]
NUMBER_OF_PLAYERS = int(getenv("NUMBER_OF_PLAYERS") or 6)
BOUNDS_COMPUTER = BOUNDS[getenv("BOUNDS_COMPUTER") or "sam_apx_1"]
APPROXIMATOR = APPROXIMATORS[getenv("APPROXIMATOR") or "max_xos"]
RNG = np.random.default_rng(getenv("SEED"))


def main() -> None:
    """Compare multiplicative algos to ours."""
    incomplete_game = IncompleteCooperativeGame(NUMBER_OF_PLAYERS, BOUNDS_COMPUTER)
    minimal_game = list(minimal_game_coalitions(NUMBER_OF_PLAYERS))
    print(minimal_game)

    print("sample", "len_queried", "approx_to_upper", "lower_to_upper_theirs", "lower_to_upper_ours")
    sample = 0
    while True:
        game = GAME_GENERATOR(NUMBER_OF_PLAYERS, RNG)

        # their approximation
        queried_coal_ids, approximated_game = APPROXIMATOR(-game)
        approximated_game = -approximated_game
        budget = len(queried_coal_ids)

        # compute our ratio
        revealed_sequence = OUR_ACTION_SEQUENCE[:budget]
        incomplete_game.set_known_values(game.get_values(minimal_game + revealed_sequence),
                                         minimal_game + revealed_sequence)
        lower_to_upper_ours = mul_factor_lower_upper_bound(incomplete_game)

        # set up incomplete game from their revealed
        queried_coals = list(set(map(Coalition, queried_coal_ids)).union(minimal_game))
        incomplete_game.set_known_values(game.get_values(queried_coals), queried_coals)

        # compute their ratios
        upper_to_approx = mul_factor_upper_to_approximation(approximated_game, incomplete_game)
        lower_to_upper_theirs = mul_factor_lower_upper_bound(incomplete_game)

        sample += 1
        print(sample, len(queried_coal_ids), upper_to_approx, lower_to_upper_theirs, lower_to_upper_ours)


if __name__ == '__main__':
    main()
