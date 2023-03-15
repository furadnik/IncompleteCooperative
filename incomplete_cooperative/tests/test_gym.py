from incomplete_cooperative.icg_gym import ICG_Gym
from incomplete_cooperative.game import IncompleteCooperativeGame
from unittest import TestCase


def dummy_bounds(game: IncompleteCooperativeGame) -> None:
    """Compute dummy bounds."""
    for coalition in filter(lambda x: x != 0, game.coalitions):
        if game.is_value_known(coalition):
            continue
        game.set_upper_bound(coalition, 1)
        game.set_lower_bound(coalition, 0)


def trivial_fill(game: IncompleteCooperativeGame) -> None:
    """Trivially fill game."""
    for coalition in game.coalitions:
        game.set_value(coalition, game.get_coalition_size(coalition))


class TestICGGym(TestCase):

    def setUp(self):
        self.game = IncompleteCooperativeGame(6, dummy_bounds)
        self.known_coalitions = [1, 2, 4, 8, 16, 32, 63]  # minimal game
        self.full_game = IncompleteCooperativeGame(6, dummy_bounds)
        trivial_fill(self.full_game)
        self.gym = ICG_Gym(self.game, self.full_game, self.known_coalitions)

    def test_zero_always_known(self):
        self.assertNotIn(0, self.gym.explorable_coalitions)
        known_coalitions = self.known_coalitions + [0]
        gym = ICG_Gym(self.game, self.full_game, known_coalitions)
        self.assertEqual(self.gym.explorable_coalitions, gym.explorable_coalitions)

    def test_explorable_coalitions(self):
        for coalition in self.known_coalitions:
            self.assertNotIn(coalition, self.gym.explorable_coalitions)

        for coalition in self.gym.explorable_coalitions:
            self.assertTrue(any([coalition in self.gym.explorable_coalitions,
                                 coalition in self.known_coalitions]))

    def test_game_setup(self):
        for coalition in self.known_coalitions:
            self.assertTrue(self.game.is_value_known(coalition))

        for coalition in self.gym.explorable_coalitions:
            self.assertFalse(self.game.is_value_known(coalition), coalition)

    def test_game_normalized(self):
        pass  # TODO: implement later.

    def test_action_mask(self):
        self.assertEqual(len(self.gym.explorable_coalitions),
                         len(self.gym.valid_action_mask))

        self.game.reveal_value(3, 3)
        mask = list(self.gym.valid_action_mask)
        for i in range(len(self.gym.explorable_coalitions)):
            if self.gym.explorable_coalitions[i] == 3:
                self.assertEqual(mask[i], 0)
                continue
            self.assertEqual(mask[i], 1, self.gym.explorable_coalitions[i])

    def test_step(self):
        self.gym.step(2)
        mask = list(self.gym.valid_action_mask)
        for i in range(len(self.gym.explorable_coalitions)):
            if i == 2:
                self.assertEqual(mask[i], 0)
                continue
            self.assertEqual(mask[i], 1, self.gym.explorable_coalitions[i])

        self.assertEqual(self.game.get_value(self.gym.explorable_coalitions[2]),
                         self.full_game.get_value(self.gym.explorable_coalitions[2]))

    def test_steps_all(self):
        for i, coalition in enumerate(self.gym.explorable_coalitions):
            with self.subTest(i=i, coalition=coalition):
                self.assertFalse(self.game.is_value_known(coalition))
                self.assertEqual(self.gym.valid_action_mask[i], 1)
                self.gym.step(i)
                self.assertTrue(self.game.is_value_known(coalition))
                self.assertEqual(self.gym.valid_action_mask[i], 0)
                self.assertEqual(self.game.get_value(coalition),
                                 self.full_game.get_value(coalition))
