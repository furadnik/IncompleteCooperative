"""Check for what percentage of games is the divergence supermodular."""
import sys

from incomplete_cooperative.bounds import compute_bounds_superadditive
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.generators import convex_generator
from incomplete_cooperative.meta_game import MetaGame
from incomplete_cooperative.norms import l1_norm
from incomplete_cooperative.supermodularity_check import check_supermodularity

divergence = l1_norm


def main(number_of_players: int) -> None:
    """Repeatedly check the divergence for supermodularity."""
    i = 0
    supermodular = 0
    while True:
        i += 1
        game = convex_generator(number_of_players)
        incomplete = IncompleteCooperativeGame(number_of_players, compute_bounds_superadditive)
        meta_game = MetaGame(game, incomplete, divergence)
        res = check_supermodularity(meta_game)
        if res is None:
            supermodular += 1
        else:
            print(*res)
        if i % 10 == 0:
            print("Samples:", i, "Supermodular:", supermodular, "Percentage:", 100 * supermodular / i)


if __name__ == '__main__':
    main(int(sys.argv[0]))
