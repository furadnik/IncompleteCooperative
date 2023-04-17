from unittest import TestCase

from incomplete_cooperative.bounds import compute_bounds_superadditive
from incomplete_cooperative.coalitions import (Coalition, all_coalitions,
                                               grand_coalition)
from incomplete_cooperative.game import IncompleteCooperativeGame


def dummy_fill(game: IncompleteCooperativeGame) -> None:
    """Compute dummy values."""
    for coalition in all_coalitions(game):
        game.set_value(len(coalition), coalition)


class TestSuperAdditiveBounds(TestCase):

    def setUp(self) -> None:
        self.game = IncompleteCooperativeGame(6, compute_bounds_superadditive)

    def test_full_game(self):
        dummy_fill(self.game)
        self.game.compute_bounds()
        for coalition in all_coalitions(self.game):
            self.assertEqual(self.game.get_value(coalition), self.game.get_upper_bound(coalition))
            self.assertEqual(self.game.get_value(coalition), self.game.get_lower_bound(coalition))

    def test_minimal_game(self):
        self.game.set_value(2 * self.game.number_of_players, grand_coalition(self.game))
        for player in range(self.game.number_of_players):
            self.game.set_value(1, Coalition.from_players([player]))
        self.game.compute_bounds()

        for coalition in filter(self.game.is_value_known, all_coalitions(self.game)):
            self.assertEqual(self.game.get_upper_bound(coalition), self.game.get_value(coalition))
            self.assertEqual(self.game.get_lower_bound(coalition), self.game.get_value(coalition))

        for coalition in filter(lambda x: not self.game.is_value_known(x), all_coalitions(self.game)):
            with self.subTest(coalition=coalition):
                self.assertEqual(self.game.get_upper_bound(coalition), self.game.number_of_players + len(coalition),
                                 self.game.get_upper_bounds())
                self.assertEqual(self.game.get_lower_bound(coalition), len(coalition),
                                 self.game.get_lower_bounds())
