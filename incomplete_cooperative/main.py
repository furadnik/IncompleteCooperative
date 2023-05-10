# type: ignore
import random
from functools import partial
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from .bounds import compute_bounds_superadditive
from .coalitions import Coalition, all_coalitions
from .game import IncompleteCooperativeGame
from .generators import generate_factory
from .icg_gym import ICG_Gym
from .protocols import Game

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


set_random_seed(42)

# Parallel environments


def masked_env():
    players = 5
    initially_known = chain([Coalition.from_players(range(players))],
                            map(lambda x: Coalition.from_players([x]), range(players)))
    gym = ICG_Gym(IncompleteCooperativeGame(players, compute_bounds_superadditive),
                  partial(generate_factory, players, 0), initially_known)
    env = ActionMasker(gym, ICG_Gym.valid_action_mask)
    return env


envs = make_vec_env(lambda: masked_env(), n_envs=4)

model = MaskablePPO(MaskableActorCriticPolicy, envs, verbose=10)
model.learn(total_timesteps=50_000)
model.save("masked_ppo")
del model  # remove to demonstrate saving and loading
model = PPO.load("masked_ppo")

episodes = 5
eval_envs = [masked_env() for _ in range(64)]
rewards = np.zeros((64, episodes))
for i, env in enumerate(eval_envs):
    obs = env.reset()
    for t in range(episodes):
        actions, _ = model.predict(obs, action_masks=env.valid_action_mask())
        obs, reward, done, info = env.step(actions)
        rewards[i, t] = reward
        if done:
            break

episode = np.arange(1, 6)
plt.grid(alpha=0.25)
plt.errorbar(episode, np.mean(rewards, axis=0), yerr=np.std(rewards, axis=0), marker='.', linestyle='')
plt.xlabel('Episode')
plt.ylabel('Mean Exploitability')
plt.savefig('average_rewards.png')
