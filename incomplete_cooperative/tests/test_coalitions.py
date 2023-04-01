from unittest import TestCase

import numpy as np

from incomplete_cooperative.coalitions import Coalition, player_to_coalition


class TestCoalitions(TestCase):

    def test_players_to_coalition(self):
        self.assertEqual(self.game_empty.players_to_coalition([0, 1, 2]),
                         0b111)
        self.assertEqual(self.game_empty.players_to_coalition([2]), 0b100)
        self.assertEqual(self.game_empty.players_to_coalition([]), 0)
        self.assertRaises(AttributeError, self.game_empty.players_to_coalition, [66])

    def test_coalition_to_players(self):
        self.assertEqual(list(self.game_empty.coalition_to_players(0b111)), [0, 1, 2])
        self.assertEqual(list(self.game_empty.coalition_to_players(0b100)), [2])
        self.assertEqual(list(self.game_empty.coalition_to_players(0)), [])
        # self.assertRaises(AttributeError, list,  # check gets called after getting the first element  # TODO: implement later.
