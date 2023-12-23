import random
from unittest import TestCase

import numpy as np

from incomplete_cooperative.bounds import compute_bounds_superadditive
from incomplete_cooperative.coalitions import (Coalition, all_coalitions,
                                               get_k_zero)
from incomplete_cooperative.exploitability import compute_exploitability
from incomplete_cooperative.generators import convex_generator
from incomplete_cooperative.meta_game import MetaGame
from incomplete_cooperative.protocols import MutableIncompleteGame
from incomplete_cooperative.supermodularity_check import check_supermodularity


class TestMetaGame(TestCase):

    def get_meta_game(self, game: MutableIncompleteGame | None = None) -> MetaGame:
        game = game if game is not None else convex_generator(4)
        game._bounds_computer = compute_bounds_superadditive  # type: ignore[attr-defined]
        return MetaGame(game, game, compute_exploitability)

    def test_meta_players(self):
        for players in range(4, 10):
            self.assertEqual(
                self.get_meta_game(convex_generator(players)).number_of_players,
                2**players - 2 - players)

    def test_get_value(self):
        for players in range(4, 10):
            game = convex_generator(players)
            game._bounds_computer = compute_bounds_superadditive
            meta_game = self.get_meta_game(game.copy())
            coalitions = [
                Coalition.from_players([1, 2]),
                Coalition.from_players([1, 3]),
                Coalition.from_players([1, 2, 3]),
                Coalition.from_players([1]),
            ] + list(get_k_zero(game))
            game.set_known_values(game.get_values(coalitions), coalitions)
            game.compute_bounds()
            self.assertEqual(compute_exploitability(game), meta_game.get_value(
                Coalition.from_players([i for i, x in enumerate(meta_game.players) if x in coalitions]))
            )

    def test_get_values(self):
        game = convex_generator(4)
        game._bounds_computer = compute_bounds_superadditive
        meta_game = self.get_meta_game(game.copy())
        meta_coalitions = [
            random.choice(list(all_coalitions(meta_game)))  # nosec
            for i in range(3)
        ]
        self.assertTrue(np.all(
            np.array([meta_game.get_value(c) for c in meta_coalitions]) == meta_game.get_values(meta_coalitions)))

    def test_check_supermodularity(self):
        self.assertIsNone(check_supermodularity(convex_generator(10)))
        self.assertIsNone(check_supermodularity(self.get_meta_game()))
