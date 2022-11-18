from game import Incomplete_Cooperative_Game

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib.pyplot as plt

set_random_seed(42)

# Parallel environments
envs = make_vec_env(lambda: Incomplete_Cooperative_Game(), n_envs=4)

model = PPO("MlpPolicy", envs, verbose=10)
model.learn(total_timesteps=50_000)
model.save("ppo")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo")

eval_envs = [Incomplete_Cooperative_Game() for _ in range(64)]
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

















