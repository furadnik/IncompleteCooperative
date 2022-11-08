from game import Incomplete_Cooperative_Game

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Parallel environments
env = make_vec_env(lambda: Incomplete_Cooperative_Game(), n_envs=4)

model = PPO("MlpPolicy", env, verbose=10)
model.learn(total_timesteps=250000)
model.save("ppo")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo")

obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    dones

















