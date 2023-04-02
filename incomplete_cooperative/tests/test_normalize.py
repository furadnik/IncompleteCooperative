from unittest import TestCase

from incomplete_cooperative.coalitions import (all_coalitions, grand_coalition,
                                               player_to_coalition)
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.normalize import normalize_game


def fill(game: IncompleteCooperativeGame, q=2) -> None:
    """Fill game trivially."""
    for coalition in all_coalitions(game):
        game.set_value(len(coalition)**q, coalition)


class TestNormalize(TestCase):

    def setUp(self):
        self.game = IncompleteCooperativeGame(6, lambda x: None)
        fill(self.game)

    def test_normalize_range(self):
        normalize_game(self.game)
        for coalition in all_coalitions(self.game):
            with self.subTest(coalition=coalition):
                self.assertLessEqual(self.game.get_value(coalition), 1)
                self.assertGreaterEqual(self.game.get_value(coalition), 0)

    def test_singletons_zero(self):
        normalize_game(self.game)
        for player in range(self.game.number_of_players):
            coalition = player_to_coalition(player)
            with self.subTest(coalition=coalition):
                self.assertEqual(self.game.get_value(coalition), 0)

    def test_grand_coalition_is_zero(self):
        fill(self.game, 1)
        normalize_game(self.game)
        self.assertEqual(self.game.get_value(grand_coalition(self.game)), 0)
