from unittest import TestCase

from stable_baselines3.common.env_checker import check_env

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.icg_gym import ICG_Gym
from incomplete_cooperative.normalize import normalize_game

from .utils import GymMixin


class TestICGGym(TestCase, GymMixin):

    def setUp(self):
        self.icg_gym = self.get_gym()
        self.game = self.icg_gym.incomplete_game
        self.known_coalitions = self.icg_gym.initially_known_coalitions  # minimal game
        self.full_game = self.icg_gym.full_game

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
                state, _, done, _, _ = self.icg_gym.step(i)
                # self.assertFalse(done) if i < len(self.icg_gym.explorable_coalitions) - 1 else self.assertTrue(done)
                self.assertTrue(self.game.is_value_known(coalition))
                self.assertEqual(state[i], self.game.get_value(coalition))
                self.assertEqual(self.icg_gym.valid_action_mask()[i], 0)
                self.assertEqual(self.game.get_value(coalition),
                                 self.full_game.get_value(coalition))

    def test_done_after_steps(self):
        icg_gym = ICG_Gym(self.game, lambda: self.full_game, self.known_coalitions, 42)
        self.assertFalse(icg_gym.done)
        for i in range(42):
            icg_gym.step(1)

        self.assertTrue(icg_gym.done)

        icg_gym.reset()
        self.assertFalse(icg_gym.done)

    def test_gym_env(self):
        check_env(self.icg_gym)

    def test_step_unstep(self):
        test_functions = [
            ("steps_taken", lambda x: x.steps_taken),
            ("reward", lambda x: x.reward.tolist()),
            ("state", lambda x: x.state.tolist()),
        ]
        for i, coalition in enumerate(self.icg_gym.explorable_coalitions):
            for test_name, test in test_functions:
                with self.subTest(coalition=coalition, test=test_name):
                    initial = test(self.icg_gym)
                    self.icg_gym.step(i)
                    self.icg_gym.unstep(i)
                    self.assertEqual(initial, test(self.icg_gym))

    def test_gym_reset_original(self):
        _, info = self.icg_gym.reset()
        self.assertNotEqual(info["game"], self.icg_gym.full_game)
        normalize_game(info["game"])
        self.assertEqual(info["game"], self.icg_gym.full_game)
