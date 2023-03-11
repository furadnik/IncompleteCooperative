from unittest import TestCase
from incomplete_cooperative.game import Incomplete_Cooperative_Game

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker


class TestGame(TestCase):

    def test_sample(self):
        print("sample test")
