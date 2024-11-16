"""Compute bounds for an incomplete game."""
from typing import Iterator

import numpy as np

from .coalitions import Coalition, all_coalitions, grand_coalition
from .functoolz import powerset
from .protocols import BoundableIncompleteGame


def _sub_coalitions(coalition: Coalition, game: BoundableIncompleteGame) -> Iterator[Coalition]:
    """Generate sub-coalitions for a coalition."""
    return (coalition.from_players(players) for players in powerset(list(coalition.players)))


def _super_coalitions(coalition: Coalition, game: BoundableIncompleteGame) -> Iterator[Coalition]:
    """Generate super-coalitions of a coalition."""
    opposite = grand_coalition(game) - coalition
    return (coalition | sub for sub in _sub_coalitions(opposite, game))


def compute_bounds_superadditive(game: BoundableIncompleteGame) -> None:
    """Compute the bounds given a superadditive incomplete game."""
    for coalition in sorted(filter(lambda x: not game.is_value_known(x), all_coalitions(game)), key=len):
        sub_coalitions = list(_sub_coalitions(coalition, game))
        complementary_coalitions = list(map(lambda x: coalition - x, sub_coalitions))
        lower_bound = np.max(game.get_lower_bounds(sub_coalitions) + game.get_lower_bounds(complementary_coalitions))
        game.set_lower_bound(lower_bound, coalition)

    for coalition in filter(lambda x: not game.is_value_known(x), all_coalitions(game)):
        known_super_coalitions = list(filter(game.is_value_known, _super_coalitions(coalition, game)))
        complementary_coalitions = list(map(lambda x: x - coalition, known_super_coalitions))
        upper_bound = np.min(game.get_values(known_super_coalitions) - game.get_lower_bounds(complementary_coalitions))
        game.set_upper_bound(upper_bound, coalition)


BOUNDS = {
    "superadditive": compute_bounds_superadditive,
}
