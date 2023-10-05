from itertools import chain, combinations
from unittest import TestCase

import numpy as np

from incomplete_cooperative.coalitions import Coalition, all_coalitions
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.graph_game import GraphCooperativeGame
from incomplete_cooperative.normalize import normalize_game

from .utils import GraphGameMixin


class TestGraphCG(GraphGameMixin, TestCase):

    def test_number_of_players(self):
        for i in range(1, 30):
            self.assertEqual(self.get_game(n_players=i).number_of_players, i)

    def test_value_singletons_zero(self):
        for i in range(1, 30):
            game = self.get_game(n_players=i)
            for j in range(i):
                coal = Coalition.from_players([j])
                self.assertEqual(game.get_value(coal), 0, game)

    def test_value_empty_zero(self):
        for i in range(1, 30):
            game = self.get_game(n_players=i)
            coal = Coalition.from_players([])
            self.assertEqual(game.get_value(coal), 0, game)

    def test_get_values(self):
        for n in range(1, 5):
            game = self.get_game(n_players=n)
            for coals in chain.from_iterable(combinations(all_coalitions(game), r) for r in range(n + 1)):
                for i, coal in enumerate(coals):
                    self.assertEqual(game.get_value(coal), game.get_values(coals)[i], game)

    def test_equality(self):
        game = self.get_game()
        other = GraphCooperativeGame(game._graph_matrix)
        self.assertEqual(game, other, game)

    def test_not_equal(self):
        game = self.get_game()
        other = self.get_game()
        self.assertNotEqual(game, other, game)

    def test_not_equal_diff_players(self):
        game = self.get_game(n_players=66)
        other = self.get_game(n_players=4)
        self.assertNotEqual(game, other, game)

    def test_not_equal_on_type(self):
        self.assertRaises(AttributeError, lambda: self.get_game() == "foobar")

    def test_not_equal_to_other_game(self):
        other = IncompleteCooperativeGame(4, lambda x: None)
        other.set_values(np.full(16, 0.0), all_coalitions(other))
        self.assertNotEqual(self.get_game(), other)

    def test_equal_to_other_game(self):
        game = self.get_game()
        other = IncompleteCooperativeGame(4, lambda x: None)
        other.set_values(game.get_values(), all_coalitions(other))
        self.assertEqual(game, other)

    def test_copy(self):
        for players in range(3, 60):
            game = self.get_game(n_players=players)
            other = game.copy()
            self.assertEqual(game, other, game)
            normalize_game(other)
            self.assertNotEqual(game, other, game)

    def test_add_ok(self):
        for players in range(3, 10):
            game = self.get_game(n_players=players)
            other = self.get_game(n_players=players)
            game_v = game.get_values()
            other_v = other.get_values()
            added = game + other
            for value, other_value, orig, orig_o, added_value in zip(
                    game.get_values(), other.get_values(), game_v, other_v, added.get_values()):
                self.assertAlmostEqual(value, orig)
                self.assertAlmostEqual(other_value, orig_o)
                self.assertAlmostEqual(added_value, orig + other_value)
