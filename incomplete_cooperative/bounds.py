"""Compute bounds for an incomplete game."""
import numpy as np

from .coalitions import (all_coalitions, get_sub_coalitions,
                         get_super_coalitions, grand_coalition)
from .protocols import BoundableIncompleteGame


def compute_bounds_superadditive(game: BoundableIncompleteGame) -> None:
    """Compute the bounds given a superadditive incomplete game."""
    for coalition in sorted(filter(lambda x: not game.is_value_known(x), all_coalitions(game)), key=len):
        sub_coalitions = list(get_sub_coalitions(coalition))
        complementary_coalitions = list(map(lambda x: coalition - x, sub_coalitions))
        lower_bound = np.max(game.get_lower_bounds(sub_coalitions) + game.get_lower_bounds(complementary_coalitions))
        game.set_lower_bound(lower_bound, coalition)

    for coalition in filter(lambda x: not game.is_value_known(x), all_coalitions(game)):
        known_super_coalitions = list(filter(game.is_value_known, get_super_coalitions(coalition, game.number_of_players)))
        complementary_coalitions = list(map(lambda x: x - coalition, known_super_coalitions))
        upper_bound = np.min(game.get_values(known_super_coalitions) - game.get_lower_bounds(complementary_coalitions))
        game.set_upper_bound(upper_bound, coalition)


BOUNDS = {
    "superadditive": compute_bounds_superadditive,
}
