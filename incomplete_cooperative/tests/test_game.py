from unittest import TestCase

import numpy as np

from incomplete_cooperative.coalitions import all_coalitions
from incomplete_cooperative.game import Coalition, IncompleteCooperativeGame
from incomplete_cooperative.protocols import (BoundableIncompleteGame,
                                              MutableGame, Value)


def dummy_bounds(game: BoundableIncompleteGame) -> None:
    """Compute dummy bounds."""
    game.set_upper_bounds(np.ones(2**game.number_of_players, dtype=Value))
    game.set_lower_bounds(np.zeros(2**game.number_of_players, dtype=Value))


def dummy_fill(game: MutableGame) -> None:
    """Compute dummy bounds."""
    for coalition in all_coalitions(game):
        game.set_value(len(coalition), coalition)


class TestGame(TestCase):

    def setUp(self) -> None:
        self.game = IncompleteCooperativeGame(6, dummy_bounds)

    def test_value_setting(self):
        self.assertIsNone(self.game.get_known_value(Coalition(1)))
        self.game.set_known_values([3], [Coalition(1)])
        self.assertEqual(self.game.get_value(Coalition(1)), 3)

    def test_value_resetting(self):
        self.game.set_value(1, Coalition(3))
        self.assertIsNotNone(self.game.get_known_value(Coalition(3)))
        self.game.set_known_values([1], [Coalition(2)])
        self.assertIsNone(self.game.get_known_value(Coalition(1)))
        self.assertEqual(self.game.get_known_value(Coalition(2)), 1)

    def test_reveal_proper(self):
        self.game.set_value(4, Coalition(2))
        self.game.reveal_value(3, Coalition(1))
        self.assertEqual(self.game.get_value(Coalition(1)), 3)
        self.assertEqual(self.game.get_value(Coalition(2)), 4)

    def test_compute_bounds(self):
        self.game.compute_bounds()
        self.assertEqual(self.game.get_lower_bound(Coalition(3)), 0)
        self.assertEqual(self.game.get_upper_bound(Coalition(3)), 1)
        self.assertEqual(list(self.game.get_interval(Coalition(3))), [0, 1])

    def test_known_values(self):
        self.assertFalse(np.any(self.game.are_values_known()[1:]))
        self.assertTrue(self.game.are_values_known()[0])
        self.game.reveal_value(1, Coalition(1))
        self.assertTrue(self.game.are_values_known()[0])
        self.assertTrue(self.game.are_values_known()[1])
        self.assertFalse(self.game.are_values_known()[4])
        self.assertTrue(np.any(self.game.are_values_known()))

    def test_get_unknown_value(self):
        self.assertRaises(ValueError, self.game.get_value, Coalition(1))

    def test_get_values_all_known(self):
        dummy_fill(self.game)
        self.assertListEqual(list(self.game.get_values()),
                             list(len(Coalition(i)) for i in range(2**self.game.number_of_players)))

    def test_get_values_some_unknown(self):
        self.assertRaises(ValueError, self.game.get_values)
        self.assertEqual(self.game.get_values([Coalition(0)]), [0])

    def test_set_all_values(self) -> None:
        dummy_fill(self.game)
        new = IncompleteCooperativeGame(self.game.number_of_players, lambda x: None)
        new.set_values(self.game.get_values())
        self.assertTrue(np.all(self.game.get_values() == new.get_values()))

    def test_get_intervals_limited(self) -> None:
        self.assertTrue(
            np.all(self.game.get_intervals([Coalition(1)]) == [self.game.get_interval(Coalition(1))]))

    def test_get_intervals_all(self) -> None:
        self.assertTrue(
            np.all(self.game.get_intervals() == [
                self.game.get_interval(Coalition(c)) for c in range(2**self.game.number_of_players)]))

    def test_get_known_values_in_full_game(self):
        dummy_fill(self.game)
        self.assertTrue(np.all(
            self.game.get_values() == self.game.get_known_values()))

    def test_get_known_values_in_partial_game(self):
        self.game.reveal_value(1, Coalition(1))
        self.game.reveal_value(5, Coalition(2))
        self.game.reveal_value(6, Coalition(4))
        self.game.reveal_value(7, Coalition(7))
        self.game.reveal_value(8, Coalition(11))
        self.game.reveal_value(9, Coalition(14))
        self.game.reveal_value(10, Coalition(17))
        for i, value in enumerate(self.game.get_known_values()):
            with self.subTest(i=i):
                if i in [0, 1, 2, 4, 7, 11, 14, 17]:
                    self.assertEqual(value, self.game.get_value(Coalition(i)))
                else:
                    self.assertTrue(np.isnan(value))

    def test_set_selective_upper_bounds(self):
        singletons = [Coalition.from_players([i]) for i in range(self.game.number_of_players)]
        dummy_bounds(self.game)
        self.assertTrue(np.all(self.game.get_upper_bounds(singletons) == 1))
        self.game.set_upper_bounds(np.full(self.game.number_of_players, .5),
                                   singletons)
        self.assertTrue(np.all(self.game.get_upper_bounds(singletons) == .5),
                        msg=self.game.get_upper_bounds(singletons))
        not_singletons = filter(lambda x: len(x) > 1, all_coalitions(self.game))
        self.assertTrue(np.all(self.game.get_upper_bounds(not_singletons) == 1))
        self.assertEqual(self.game.get_upper_bound(Coalition(0)), 0)

    def test_set_single_upper_bound(self):
        dummy_bounds(self.game)
        self.assertTrue(np.all(self.game.get_upper_bounds(filter(lambda x: len(x), all_coalitions(self.game))) == 1))
        self.game.set_upper_bound(.5, Coalition(3))
        self.assertTrue(np.all(self.game.get_upper_bounds(filter(lambda x: x != Coalition(3) and len(x),
                                                                 all_coalitions(self.game))) == 1))
        self.assertTrue(np.all(self.game.get_upper_bounds(filter(lambda x: x == Coalition(3),
                                                                 all_coalitions(self.game))) == .5))
        self.assertEqual(self.game.get_upper_bound(Coalition(3)), .5)
        self.assertEqual(self.game.get_upper_bound(Coalition(0)), 0)

    def test_set_selective_lower_bounds(self):
        singletons = [Coalition.from_players([i]) for i in range(self.game.number_of_players)]
        dummy_bounds(self.game)
        self.assertTrue(np.all(self.game.get_lower_bounds(singletons) == 0))
        self.game.set_lower_bounds(np.full(self.game.number_of_players, .5),
                                   singletons)
        self.assertTrue(np.all(self.game.get_lower_bounds(singletons) == .5),
                        msg=self.game.get_lower_bounds(singletons))
        not_singletons = filter(lambda x: len(x) != 1, all_coalitions(self.game))
        self.assertTrue(np.all(self.game.get_lower_bounds(not_singletons) == 0))
        self.assertEqual(self.game.get_lower_bound(Coalition(0)), 0)

    def test_set_single_lower_bound(self):
        dummy_bounds(self.game)
        self.assertTrue(np.all(self.game.get_lower_bounds(filter(lambda x: len(x), all_coalitions(self.game))) == 0))
        self.game.set_lower_bound(.5, Coalition(3))
        self.assertTrue(np.all(self.game.get_lower_bounds(filter(lambda x: x != Coalition(3),
                                                                 all_coalitions(self.game))) == 0))
        self.assertTrue(np.all(self.game.get_lower_bounds(filter(lambda x: x == Coalition(3),
                                                                 all_coalitions(self.game))) == .5))
        self.assertEqual(self.game.get_lower_bound(Coalition(3)), .5)

    def test_compare_games(self):
        self.assertEqual(self.game, IncompleteCooperativeGame(self.game.number_of_players, lambda x: None))
        dummy_fill(self.game)
        self.assertNotEqual(self.game, IncompleteCooperativeGame(self.game.number_of_players, lambda x: None))
        self.assertRaises(AttributeError, lambda: self.game == 42)

    def test_is_game_full(self):
        self.assertFalse(self.game.full)
        dummy_fill(self.game)
        self.assertTrue(self.game.full)

    def test_game_value_set_unset(self):
        self.game.set_value(10, Coalition(1))
        self.assertEqual(self.game.get_value(Coalition(1)), 10)
        self.game.unset_value(Coalition(1))
        self.assertRaises(ValueError, self.game.get_value, Coalition(1))
        self.assertFalse(self.game.is_value_known(Coalition(1)))

    def test_game_value_unreveal(self):
        self.game.reveal_value(10, Coalition(1))
        self.assertEqual(self.game.get_value(Coalition(1)), 10)
        self.game.unreveal_value(Coalition(1))
        self.assertRaises(ValueError, self.game.get_value, Coalition(1))
        self.assertFalse(self.game.is_value_known(Coalition(1)))

    def test_copy(self):
        game = self.game.copy()
        game.set_value(100, Coalition(1))
        self.assertFalse(self.game.is_value_known(Coalition(1)))
        self.assertTrue(game.is_value_known(Coalition(1)))

    def test_add_ok(self):
        dummy_fill(self.game)
        game = self.game.copy()
        other = self.game.copy()
        added = game + other
        for value, other_value, orig, added_value in zip(
                game.get_values(), other.get_values(), self.game.get_values(), added.get_values()):
            self.assertEqual(value, orig)
            self.assertEqual(other_value, orig)
            self.assertEqual(added_value, 2 * orig)

    def test_negation(self):
        game = IncompleteCooperativeGame(5)
        game.set_known_values([1, 2, 3], [Coalition(1), Coalition(2), Coalition(3)])
        initial = game.get_known_values()
        ngame = -game
        assert np.all(ngame.get_known_values()[1 - np.isnan(initial)] == -initial[1 - np.isnan(initial)])
        assert np.all(ngame.are_values_known() == game.are_values_known())
        assert np.all(ngame.are_values_known() == 1 - np.isnan(initial))
        assert np.all(game.get_known_values()[1 - np.isnan(initial)] == initial[1 - np.isnan(initial)])
