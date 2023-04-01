from unittest import TestCase

import numpy as np

from incomplete_cooperative.coalitions import all_coalitions
from incomplete_cooperative.game import Coalition, IncompleteCooperativeGame
from incomplete_cooperative.protocols import (BoundableIncompleteGame,
                                              MutableIncompleteGame)


def dummy_bounds(game: BoundableIncompleteGame) -> None:
    """Compute dummy bounds."""
    for coalition in filter(lambda x: x != 0, all_coalitions(game)):
        game.set_upper_bound(1, coalition)
        game.set_lower_bound(0, coalition)


class TestGame(TestCase):

    def setUp(self) -> None:
        self.game_empty = IncompleteCooperativeGame(6, dummy_bounds)

    def test_implements_protocols(self):
        self.assertIsInstance(self.game_empty, MutableIncompleteGame)
        self.assertIsInstance(self.game_empty, BoundableIncompleteGame)

    def test_value_setting(self):
        self.assertIsNone(self.game_empty.get_value(Coalition(1)))
        self.game_empty.set_known_values([3], [Coalition(1)])
        self.assertEqual(self.game_empty.get_value(Coalition(1)), 3)

    def test_value_resetting(self):
        self.game_empty.set_value(Coalition(3), 1)
        self.assertIsNotNone(self.game_empty.get_known_value(Coalition(3)))
        self.game_empty.set_known_values([1], [Coalition(2)])
        self.assertIsNone(self.game_empty.get_known_value(Coalition(1)))
        self.assertEqual(self.game_empty.get_known_value(Coalition(2)), 1)

    def test_reveal_existing(self):
        self.game_empty.set_value(1, Coalition(3))
        self.assertRaises(ValueError, self.game_empty.reveal_value, 1, Coalition(3))
        self.assertEqual(self.game_empty.get_value(1), 3)

    def test_reveal_proper(self):
        self.game_empty.set_value(2, 4)
        self.game_empty.reveal_value(1, 3)
        self.assertEqual(self.game_empty.get_value(1), 3)
        self.assertEqual(self.game_empty.get_value(2), 4)

    def test_compute_bounds(self):
        self.game_empty.compute_bounds()
        self.assertEqual(self.game_empty.get_lower_bound(3), 0)
        self.assertEqual(self.game_empty.get_upper_bound(3), 1)
        self.assertEqual(list(self.game_empty.get_bounds(3)), [0, 1])

    def test_known_values(self):
        self.assertFalse(np.any(self.game_empty.known_values[1:]))
        self.assertTrue(self.game_empty.known_values[0])
        self.game_empty.reveal_value(1, 1)
        self.assertTrue(self.game_empty.known_values[0])
        self.assertTrue(self.game_empty.known_values[1])
        self.assertFalse(self.game_empty.known_values[4])
        self.assertTrue(np.any(self.game_empty.known_values))

    def coalition_filtering_function_tester(self, condition, coalitions):
        coalitions = list(coalitions)
        complement = [x for x in self.game_empty.coalitions if x not in coalitions]
        for coalition in coalitions:
            self.assertTrue(condition(coalition), coalition)
        for coalition in complement:
            self.assertFalse(condition(coalition))

    def test_filter_include(self):
        self.coalition_filtering_function_tester(
            lambda x: 2 in self.game_empty.coalition_to_players(x) and 0 in self.game_empty.coalition_to_players(x),
            self.game_empty.filter_coalitions_include_coalition(5, self.game_empty.coalitions)
        )

    def test_filter_include_some(self):
        self.coalition_filtering_function_tester(
            lambda x: 2 in self.game_empty.coalition_to_players(x) or 0 in self.game_empty.coalition_to_players(x),
            self.game_empty.filter_coalitions_include_some_coalition(5, self.game_empty.coalitions)
        )

    def test_filter_not_include(self):
        self.coalition_filtering_function_tester(
            lambda x:
                2 not in self.game_empty.coalition_to_players(x) and 0 not in self.game_empty.coalition_to_players(x),
            self.game_empty.filter_coalitions_not_include_coalition(5, self.game_empty.coalitions)
        )

    def test_filter_empty(self):
        self.assertEqual(list(self.game_empty.filter_coalitions_not_include_coalition(0, self.game_empty.coalitions)),
                         list(self.game_empty.coalitions))

    def test_filter_full(self):
        self.assertEqual(list(self.game_empty.filter_coalitions_include_coalition(0, self.game_empty.coalitions)), [])

    def test_filter_subsets(self):
        self.coalition_filtering_function_tester(
            lambda x:
                2 in self.game_empty.coalition_to_players(x) and 0 in self.game_empty.coalition_to_players(x),
            self.game_empty.filter_coalition_subset(5, self.game_empty.coalitions)
        )

    def test_filter_subsets_proper(self):
        self.coalition_filtering_function_tester(
            lambda x: all([2 in self.game_empty.coalition_to_players(x),
                           0 in self.game_empty.coalition_to_players(x),
                           list(filter(lambda x: x not in [0, 2], self.game_empty.coalition_to_players(x))) != []]),
            self.game_empty.filter_coalition_subset(5, self.game_empty.coalitions, proper=True)
        )

    def test_coalition_size(self):
        for coalition in self.game_empty.coalitions:
            with self.subTest(coalition=coalition):
                self.assertEqual(len(list(self.game_empty.coalition_to_players(coalition))),
                                 self.game_empty.get_coalition_size(coalition))

    def test_get_bounds(self):
        self.game_empty.set_known_values({
            1: 1,
            2: 2
        })
        self.assertEqual(self.game_empty.values[1], 1)
        self.assertEqual(self.game_empty.values[2], 2)
        self.assertEqual(self.game_empty.values[3], False)

    def test_full(self):
        self.assertFalse(self.game_empty.full)
        for coalition in self.game_empty.coalitions:
            self.game_empty.set_value(coalition, self.game_empty.get_coalition_size(coalition))
        self.assertTrue(self.game_empty.full)

    def test_bounds(self):
        dummy_bounds(self.game_empty)
        self.assertEqual(np.sum(self.game_empty.lower_bounds), 0)
        self.assertEqual(np.sum(self.game_empty.get_upper_bounds), 2**self.game_empty.number_of_players - 1)

    def test_set_bound_known(self):
        self.assertRaises(AttributeError,
                          self.game_empty.set_lower_bound, 0, 1)
        self.assertRaises(AttributeError,
                          self.game_empty.set_upper_bound, 0, 1)
