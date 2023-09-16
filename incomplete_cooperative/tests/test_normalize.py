from unittest import TestCase

import numpy as np

from incomplete_cooperative.coalitions import (all_coalitions, grand_coalition,
                                               player_to_coalition)
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.normalize import normalize_game

from .utils import GraphGameMixin


def fill(game: IncompleteCooperativeGame, q=2) -> None:
    """Fill game trivially."""
    for coalition in all_coalitions(game):
        game.set_value(len(coalition)**q, coalition)


class TestNormalizeIncompleteGame(TestCase):

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


class TestNormalizeGraphGame(GraphGameMixin, TestCase):

    def test_grand_coalition_one(self):
        for i in range(2, 100):
            game = self.get_game(n_players=i)
            self.assertNotEqual(game.get_value(grand_coalition(game)), 1)
            normalize_game(game)
            self.assertAlmostEqual(game.get_value(grand_coalition(game)), 1, 5, msg=game)

    def test_graph_equal_incomplete(self):
        for i in range(2, 12):
            game = self.get_game(n_players=i)
            incomplete = self.to_incomplete(game)
            normalize_game(game)
            normalize_game(incomplete)
            self.assertAlmostEqual(np.max(np.abs(game.get_values() - incomplete.get_values())), 0, 6, game)

    def test_graph_zero_game(self):
        for i in range(2, 12):
            game = self.get_game(n_players=i)
            game._graph_matrix = np.zeros_like(game._graph_matrix)
            normalize_game(game)
            self.assertAlmostEqual(game.get_value(grand_coalition(game)), 0, 6, game)
