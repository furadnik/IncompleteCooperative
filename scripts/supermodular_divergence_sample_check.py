"""Check for what percentage of games is the divergence supermodular."""
import sys

from incomplete_cooperative.bounds import compute_bounds_superadditive
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.generators import convex_generator
from incomplete_cooperative.meta_game import MetaGame
from incomplete_cooperative.norms import l1_norm
from incomplete_cooperative.supermodularity_check import (
    check_failed_diagnostics, check_supermodularity)

divergence = l1_norm


def main() -> None:
    """Check the divergence for supermodularity."""
    number_of_players = int(sys.argv[1])
    game = convex_generator(number_of_players)
    incomplete = IncompleteCooperativeGame(number_of_players, compute_bounds_superadditive)
    meta_game = MetaGame(game, incomplete, divergence)
    res = check_supermodularity(meta_game)
    if res is None:
        print("supermodular")
    else:
        print("not supermodular")
        check_failed_diagnostics(meta_game, *res)


if __name__ == '__main__':
    main()
