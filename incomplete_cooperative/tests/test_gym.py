from unittest import TestCase

from incomplete_cooperative.coalitions import Coalition, all_coalitions
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.icg_gym import ICG_Gym
from incomplete_cooperative.protocols import MutableIncompleteGame


def trivial_fill(game: MutableIncompleteGame) -> None:
    """Trivially fill game."""
    for coalition in all_coalitions(game):
        game.set_value(len(coalition), coalition)


class TestICGGym(TestCase):

    def setUp(self):
        self.game = IncompleteCooperativeGame(6, lambda x: None)
        self.known_coalitions = list(map(Coalition, [1, 2, 4, 8, 16, 32, 63]))  # minimal game
        self.full_game = IncompleteCooperativeGame(6, lambda x: None)
        trivial_fill(self.full_game)
        self.icg_gym = ICG_Gym(self.game, lambda: self.full_game, self.known_coalitions)

    def test_zero_always_known(self):
        self.assertNotIn(0, self.icg_gym.explorable_coalitions)
        known_coalitions = self.known_coalitions + [Coalition(0)]
        gym = ICG_Gym(self.game, lambda: self.full_game, known_coalitions)
        self.assertEqual(self.icg_gym.explorable_coalitions, gym.explorable_coalitions)

    def test_explorable_coalitions(self):
        for coalition in self.known_coalitions:
            self.assertNotIn(coalition, self.icg_gym.explorable_coalitions)

        for coalition in self.icg_gym.explorable_coalitions:
            self.assertTrue(any([coalition in self.icg_gym.explorable_coalitions,
                                 coalition in self.known_coalitions]))

    def test_game_setup(self):
        for coalition in self.known_coalitions:
            self.assertTrue(self.game.is_value_known(coalition))

        for coalition in self.icg_gym.explorable_coalitions:
            self.assertFalse(self.game.is_value_known(coalition), coalition)

    def test_action_mask(self):
        self.assertEqual(len(self.icg_gym.explorable_coalitions),
                         len(self.icg_gym.valid_action_mask()))

        self.game.reveal_value(3, Coalition(3))
        mask = list(self.icg_gym.valid_action_mask())
        for i in range(len(self.icg_gym.explorable_coalitions)):
            with self.subTest(i=i, coalition=self.icg_gym.explorable_coalitions[i]):
                if self.icg_gym.explorable_coalitions[i] == Coalition(3):
                    self.assertFalse(mask[i])
                    continue
                self.assertTrue(mask[i], self.icg_gym.explorable_coalitions[i])

    def test_step(self):
        self.icg_gym.step(2)
        mask = list(self.icg_gym.valid_action_mask())
        for i in range(len(self.icg_gym.explorable_coalitions)):
            if i == 2:
                self.assertEqual(mask[i], 0)
                continue
            self.assertEqual(mask[i], 1, self.icg_gym.explorable_coalitions[i])

        self.assertEqual(self.game.get_value(self.icg_gym.explorable_coalitions[2]),
                         self.full_game.get_value(self.icg_gym.explorable_coalitions[2]))

    def test_steps_all(self):
        for i, coalition in enumerate(self.icg_gym.explorable_coalitions):
            with self.subTest(i=i, coalition=coalition):
                self.assertFalse(self.game.is_value_known(coalition))
                self.assertEqual(self.icg_gym.valid_action_mask()[i], 1)
                state, _, done, _ = self.icg_gym.step(i)
                # self.assertFalse(done) if i < len(self.icg_gym.explorable_coalitions) - 1 else self.assertTrue(done)
                self.assertTrue(self.game.is_value_known(coalition))
                self.assertEqual(state[i], self.game.get_value(coalition))
                self.assertEqual(self.icg_gym.valid_action_mask()[i], 0)
                self.assertEqual(self.game.get_value(coalition),
                                 self.full_game.get_value(coalition))
