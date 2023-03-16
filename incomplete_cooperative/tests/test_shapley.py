from unittest import TestCase
from incomplete_cooperative.shapley import compute_shapley_value
from incomplete_cooperative.game import IncompleteCooperativeGame


def dummy_bounds(game: IncompleteCooperativeGame) -> None:
    """Compute dummy bounds."""
    for coalition in game.coalitions:
        if game.get_value(coalition) is not None:
            continue
        game.set_upper_bound(coalition, 1)
        game.set_lower_bound(coalition, 0)


def fill_trivial(game: IncompleteCooperativeGame) -> None:
    """Fill trivially with sizes of coalitions."""
    for coalition in game.coalitions:
        game.set_value(coalition, game.get_coalition_size(coalition))


class TestShapley(TestCase):

    def setUp(self):
        self.game_empty = IncompleteCooperativeGame(6, dummy_bounds)

    def test_shapley_size(self):
        func = lambda x: x.lower_bounds
        self.game_empty.compute_bounds()
        self.assertEqual(list(compute_shapley_value(self.game_empty, func)),
                         [0] * self.game_empty.number_of_players)

    def test_shapley_incomplete(self):
        fill_trivial(self.game_empty)
        values = list(compute_shapley_value(self.game_empty))
        for val in values:
            self.assertEqual(val, values[0])

    def test_shapley_worker_factory(self):
        """There is one owner, other players are workers.

        If the coalition has an owner, then it produces `n`, the number of workers in the coalition.
        Without an owner, there is no factory, so nothing gets produced.

        The owner should get half of the overall value, as without him, nothing gets produced.
        """
        n = 10
        game = IncompleteCooperativeGame(n, dummy_bounds)
        for coalition in game.get_coalitions_not_including_players([0]):
            game.set_value(coalition, 0)
            game.set_value(coalition + 1, game.get_coalition_size(coalition))

        shapley = compute_shapley_value(game, lambda x: x.values)
        self.assertEqual(next(shapley), (n - 1) / 2)
        for i in range(1, n):
            self.assertEqual(next(shapley), .5)
