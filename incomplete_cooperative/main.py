from .gym import ICG_Gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib.pyplot as plt

set_random_seed(42)

# Parallel environments


def masked_env():
    def mask_fn(env):
        return env.valid_action_mask()

    env = ICG_Gym()
    env = ActionMasker(env, mask_fn)
    return env


envs = make_vec_env(lambda: masked_env(), n_envs=4)

model = MaskablePPO(MaskableActorCriticPolicy, envs, verbose=10)
model.learn(total_timesteps=50_000)
model.save("masked_ppo")

del model  # remove to demonstrate saving and loading

model = PPO.load("masked_ppo")

eval_envs = [masked_env() for _ in range(64)]
rewards = np.zeros((64, 5))
for i, env in enumerate(eval_envs):
    obs = env.reset()
    for t in range(5):
        actions, _states = model.predict(obs)
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
