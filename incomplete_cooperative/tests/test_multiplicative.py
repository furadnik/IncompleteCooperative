"""Test the multiplicative module."""
import numpy as np
import pytest

from incomplete_cooperative.coalitions import player_to_coalition
from incomplete_cooperative.generators import (covg_fn_generator,
                                               k_budget_generator)
from incomplete_cooperative.multiplicative.max_xos_approximation import \
    compute_max_xos_approximation

# from incomplete_cooperative.multiplicative.rla_approximation import \
#     compute_rla_approximation


@pytest.mark.parametrize("rep", range(10))
@pytest.mark.parametrize("players", range(4, 10))
@pytest.mark.parametrize("approximator_generator", [
    (compute_max_xos_approximation, covg_fn_generator),
    (compute_max_xos_approximation, k_budget_generator),
    # (compute_rla_approximation, covg_fn_generator),
    # (compute_rla_approximation, k_budget_generator),
])
def test_is_lower_bound(rep, approximator_generator, players):
    approximator, generator = approximator_generator
    game = generator(players)

    singleton_values = -game.get_values([player_to_coalition(player) for player in range(players)])
    # we need to generate a game with all singletons bigger than 1
    game.set_values(-game.get_values() / np.min(singleton_values))

    _, approx_game = approximator(game)
    assert np.all(approx_game.get_values() <= game.get_values()), np.max(approx_game.get_values() - game.get_values())
