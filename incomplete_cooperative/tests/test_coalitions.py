from itertools import chain, combinations
from unittest import TestCase

from incomplete_cooperative.coalitions import (Coalition, all_coalitions,
                                               exclude_coalition,
                                               minimal_game_coalitions,
                                               player_to_coalition)
from incomplete_cooperative.game import IncompleteCooperativeGame


class TestCoalitions(TestCase):

    def setUp(self) -> None:
        self.number_of_players = 6
        self.game = IncompleteCooperativeGame(self.number_of_players)

    def test_coalition_from_players(self):
        self.assertEqual(Coalition.from_players([0, 1, 2]).id,
                         0b111)
        self.assertEqual(Coalition.from_players([2]).id, 0b100)
        self.assertEqual(Coalition.from_players([]).id, 0)

    def test_coalition_to_players(self):
        self.assertEqual(list(Coalition(0b111).players), [0, 1, 2])
        self.assertEqual(list(Coalition(0b100).players), [2])
        self.assertEqual(list(Coalition(0).players), [])

    def test_players_to_coalition(self):
        self.assertEqual(Coalition(0b100), player_to_coalition(2))
        self.assertEqual(Coalition(0b1), player_to_coalition(0))

    def test_coalition_size(self):
        powerset = chain.from_iterable(
            map(lambda r: combinations(range(10), r), range(11))
        )
        for s in powerset:
            coalition = Coalition.from_players(s)
            self.assertEqual(len(coalition), len(s))

    def test_coalition_equal(self):
        powerset = chain.from_iterable(
            map(lambda r: combinations(range(10), r), range(11))
        )
        for s in powerset:
            coalition = Coalition.from_players(s)
            for y in powerset:
                coalition2 = Coalition.from_players(y)
                self.assertEqual(coalition == coalition2, s == y)

    def test_coal_minus(self):
        coal = Coalition.from_players([1, 2])
        self.assertEqual(coal - 3, coal)
        self.assertEqual(coal - 2, Coalition.from_players([1]))

    def test_coalition_and(self):
        coal1 = Coalition.from_players([1, 2])
        coal2 = Coalition.from_players([2, 3])
        self.assertEqual(coal1 & coal2, Coalition.from_players([2]))

    def test_coalition_and_disjoint(self):
        coal1 = Coalition.from_players([1, 2])
        coal2 = Coalition.from_players([3, 5])
        self.assertEqual(coal1 & coal2, Coalition.from_players([]))

    def coalition_filtering_function_tester(self, condition, coalitions):
        coalitions = list(coalitions)
        complement = [Coalition(x) for x in range(1, 2**self.number_of_players) if Coalition(x) not in coalitions]
        for coalition in coalitions:
            self.assertTrue(condition(coalition), coalition)
        for coalition in complement:
            self.assertFalse(condition(coalition), coalition)

    def test_exclude_coalition(self) -> None:
        for coalition_id in range(1, 2**self.number_of_players):
            coalition = Coalition(coalition_id)
            assert self.number_of_players == self.game.number_of_players
            assert len(list(all_coalitions(self.game))) == 2**self.number_of_players
            self.coalition_filtering_function_tester(
                lambda x: (x & coalition) == Coalition(0),
                exclude_coalition(coalition, all_coalitions(self.game))
            )

    def test_contains(self):
        coalition = Coalition(3)
        self.assertTrue(0 in coalition)
        self.assertTrue(1 in coalition)
        self.assertFalse(3 in coalition)

    def test_add_a_single_player(self):
        coalition = Coalition(3)
        coalition |= 2
        self.assertEqual(coalition, Coalition(7))

    def test_diff_a_single_player(self):
        coalition = Coalition(3)
        coalition &= 1
        self.assertEqual(coalition, Coalition(2))

    def test_order_unknown(self):
        self.assertFalse(Coalition(3) == "Coalition(3)")

    def test_minimal_game(self):
        tested_coalitions = list(minimal_game_coalitions(self.game))
        for coalition in all_coalitions(self.game):
            with self.subTest(coalition=coalition):
                if len(coalition) <= 1 or len(coalition) == self.game.number_of_players:
                    self.assertIn(coalition, tested_coalitions)
                else:
                    self.assertNotIn(coalition, tested_coalitions)
