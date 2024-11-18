"""Compute bounds for an incomplete game."""
from functools import cache
from typing import Any

import numpy as np

from .coalition_ids import CoalitionId
from .coalition_ids import all_coalitions as all_coalitions_id
from .coalition_ids import size
from .coalition_ids import sub_coalitions as get_sub_coalitions_id
from .coalition_ids import super_coalitions as get_super_coalitions_id
from .coalitions import (Coalition, all_coalitions, get_sub_coalitions,
                         get_super_coalitions)
from .protocols import BoundableIncompleteGame


def compute_bounds_superadditive(game: BoundableIncompleteGame) -> None:
    """Compute the bounds given a superadditive incomplete game."""
    for coalition in sorted(filter(lambda x: not game.is_value_known(x), all_coalitions(game)), key=len):
        sub_coalitions = list(get_sub_coalitions(coalition))
        complementary_coalitions = [coalition - x for x in sub_coalitions]
        lower_bound = np.max(game.get_lower_bounds(sub_coalitions) + game.get_lower_bounds(complementary_coalitions))
        game.set_lower_bound(lower_bound, coalition)

    for coalition in filter(lambda x: not game.is_value_known(x), all_coalitions(game)):
        known_super_coalitions = [x for x in get_super_coalitions(coalition, game.number_of_players)
                                  if game.is_value_known(x)]
        complementary_coalitions = [x - coalition for x in known_super_coalitions]
        upper_bound = np.min(game.get_values(known_super_coalitions) - game.get_lower_bounds(complementary_coalitions))
        game.set_upper_bound(upper_bound, coalition)


@cache
def _get_sub_super_coalition_structure(number_of_players: int) -> tuple[
    np.ndarray[Any, np.dtype[CoalitionId]],
    np.ndarray[Any, np.dtype[CoalitionId]],
    np.ndarray[Any, np.dtype[CoalitionId]]
]:
    """Get the system of subcoalitions and supercoalitions of a given coalition."""
    r = []
    for coalition in all_coalitions_id(number_of_players):
        all_coals = np.zeros(2**number_of_players)
        sub_coals = get_sub_coalitions_id(coalition, number_of_players)
        all_coals[sub_coals] = 1
        super_coals = get_super_coalitions_id(coalition, number_of_players)
        all_coals[super_coals] = 2
        all_coals[coalition] = 0
        r.append(all_coals)
    sizes = np.array([size(coal, number_of_players) for coal in all_coalitions_id(number_of_players)])
    all_sorted = all_coalitions_id(number_of_players)[np.argsort(sizes)]
    return all_coalitions_id(number_of_players), all_sorted, np.array(r)


def compute_bounds_superadditive_cached(game: BoundableIncompleteGame) -> None:
    """Compute the bounds given a superadditive incomplete game."""
    assert game.is_value_known(Coalition(0))
    assert game.is_value_known(Coalition(2**game.number_of_players - 1))
    all_coalitions, all_sorted, coal_structure = _get_sub_super_coalition_structure(game.number_of_players)
    unknown_sorted = all_sorted[np.logical_not(game.are_values_known()[all_sorted])]
    for coalition in unknown_sorted:
        sub_coalitions = all_coalitions[coal_structure[coalition] == 1]
        complementary_coalitions = coalition ^ sub_coalitions
        lower_bound = np.max(game.get_lower_bounds()[sub_coalitions] + game.get_lower_bounds()[complementary_coalitions])
        game.set_lower_bound(lower_bound, Coalition(coalition))

    for coalition in unknown_sorted:
        super_coalitions = all_coalitions[coal_structure[coalition] == 2]
        known_super_coalitions = super_coalitions[game.are_values_known()[super_coalitions]]
        complementary_coalitions = coalition ^ known_super_coalitions
        upper_bound = np.min(game.get_lower_bounds()[known_super_coalitions] - game.get_lower_bounds()[complementary_coalitions])
        game.set_upper_bound(upper_bound, Coalition(coalition))


BOUNDS = {
    "superadditive": compute_bounds_superadditive,
    "superadditive_cached": compute_bounds_superadditive_cached,
}
