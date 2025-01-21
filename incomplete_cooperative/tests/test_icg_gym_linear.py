from unittest import TestCase

from incomplete_cooperative.icg_gym_linear import ICG_Gym_Linear
from incomplete_cooperative.tests.utils import GymMixin
from sb3_contrib import MaskablePPO


class TestICGGymLinear(TestCase, GymMixin):
    def test_init(self):
        icg_gym = self.get_gym()
        icg_gym_wrapper = ICG_Gym_Linear(icg_gym)
        self.assertEqual(icg_gym, icg_gym_wrapper.icg_gym)
        self.assertEqual(icg_gym.incomplete_game.number_of_players, icg_gym_wrapper.N)

    def test_action_masks(self):
        icg_gym = self.get_gym()
        icg_gym_wrapper = ICG_Gym_Linear(icg_gym)
        self.assertEqual(icg_gym_wrapper.action_masks().shape, (icg_gym_wrapper.N,))
        self.assertEqual(icg_gym_wrapper.action_masks().dtype, bool)

    def test_cannot_reveal_coalition_of_size_one(self):
        icg_gym = self.get_gym()
        icg_gym_wrapper = ICG_Gym_Linear(icg_gym)
        self.assertEqual(icg_gym_wrapper.action_masks()[1], 0)

    def test_reset(self):
        icg_gym = self.get_gym()
        icg_gym_wrapper = ICG_Gym_Linear(icg_gym)
        state, info = icg_gym_wrapper.reset()
        self.assertEqual(state.shape, (icg_gym_wrapper.N,))
        self.assertEqual(info, {"game": icg_gym.full_game})

    def test_step(self):
        icg_gym = self.get_gym()
        icg_gym_wrapper = ICG_Gym_Linear(icg_gym)
        num_pairs = icg_gym_wrapper.N * (icg_gym_wrapper.N - 1) // 2  # There are n choose 2 coalitions of size two
        for i in range(num_pairs):
            self.assertTrue(icg_gym_wrapper.action_masks()[2],
                            f'Cannot reveal any more coalitions of size 2 after {i} steps')
            icg_gym_wrapper.step(coalition_size=2)

        # Check that we can't reveal any more coalitions
        self.assertFalse(icg_gym_wrapper.action_masks()[2])

    def test_ppo_maskable(self):
        icg_gym = self.get_gym()
        icg_gym_wrapper = ICG_Gym_Linear(icg_gym)
        model = MaskablePPO("MlpPolicy", icg_gym_wrapper, verbose=0)
        model.learn(total_timesteps=10)
        self.assertTrue(True)
