"""Compute bounds for an incomplete game."""
from typing import Iterator

import numpy as np

from .coalitions import Coalition, all_coalitions
from .protocols import BoundableIncompleteGame


def _sub_coalitions(coalition) -> Iterator[Coalition]:
    """Generate sub-coalitions for a coalition."""
    return filter(lambda x: x & coalition != coalition and x | coalition == coalition,
                  all_coalitions(coalition.players))


def compute_bounds_superadditive(game: BoundableIncompleteGame) -> None:
    """Compute the bounds given a superadditive incomplete game."""
    for coalition in sorted(filter(lambda x: not game.is_value_known(x), all_coalitions(game)), key=len):
        sub_coalitions = list(_sub_coalitions(coalition))
        complementary_coalitions = map(lambda x: coalition - x, sub_coalitions)
        lower_bound = np.max(game.get_lower_bounds(sub_coalitions) + game.get_lower_bounds(complementary_coalitions))
        game.set_lower_bound(lower_bound, coalition)

        known_sub_coalitions = list(filter(lambda x: game.is_value_known(x), sub_coalitions))
        complementary_coalitions = map(lambda x: coalition - x, known_sub_coalitions)
        upper_bound = np.min(game.get_values(known_sub_coalitions) - game.get_lower_bounds(complementary_coalitions))
        game.set_upper_bound(upper_bound, coalition)
