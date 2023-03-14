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
        func = lambda x, y: x.get_lower_bound(y)
        self.game_empty.compute_bounds()
        self.assertEqual(list(compute_shapley_value(self.game_empty, func)),
                         [0] * self.game_empty.number_of_players)

    def test_shapley_incomplete(self):
        fill_trivial(self.game_empty)
        values = list(compute_shapley_value(self.game_empty))
        for val in values:
            self.assertEqual(val, values[0])
