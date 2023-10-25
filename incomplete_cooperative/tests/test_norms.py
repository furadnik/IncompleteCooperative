"""Test the `norms` module."""
from unittest import TestCase

from incomplete_cooperative.norms import lp_norm

from .utils import IncompleteGameMixin


class TestLpNorm(TestCase, IncompleteGameMixin):

    def test_lp_norm(self):
        for number_of_players in range(3, 10):
            game = self.get_game_minimal(number_of_players)
            game.compute_bounds()
            self.assertEqual(lp_norm(game, 1), (number_of_players - 1) * (2**number_of_players - number_of_players - 2))
