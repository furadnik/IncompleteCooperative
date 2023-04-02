from unittest import TestCase

from incomplete_cooperative.coalitions import (Coalition, all_coalitions,
                                               exclude_coalition)
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.protocols import BoundableIncompleteGame
from incomplete_cooperative.shapley import compute_shapley_value


def fill_trivial(game: BoundableIncompleteGame) -> None:
    """Fill trivially with sizes of coalitions."""
    for coalition in all_coalitions(game):
        game.set_value(len(coalition), coalition)


class TestShapley(TestCase):

    def setUp(self):
        self.game_empty = IncompleteCooperativeGame(6, lambda x: None)
        fill_trivial(self.game_empty)

    def test_shapley_size(self):
        self.game_empty.compute_bounds()
        self.assertEqual(list(compute_shapley_value(self.game_empty)),
                         [1] * self.game_empty.number_of_players)

    def test_shapley_worker_factory(self):
        """There is one owner, other players are workers.

        If the coalition has an owner, then it produces `n`, the number of workers in the coalition.
        Without an owner, there is no factory, so nothing gets produced.

        The owner should get half of the overall value, as without him, nothing gets produced.
        """
        n = 10
        game = IncompleteCooperativeGame(n, lambda x: None)
        for coalition in exclude_coalition(Coalition.from_players([0]), all_coalitions(game)):
            game.set_value(0, coalition)
            game.set_value(len(coalition), coalition | 0)

        shapley = compute_shapley_value(game)
        self.assertEqual(next(shapley), (n - 1) / 2)
        for i in range(1, n):
            self.assertEqual(next(shapley), .5)
