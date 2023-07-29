"""Evaluating script."""
import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks  # type: ignore

from .model import ModelInstance


def eval_func(instance: ModelInstance, parsed_args) -> None:
    """Evaluate the model."""
    model = instance.model
    env = instance.env_generator()
    rewards_all = np.zeros((parsed_args.eval_episode_length, parsed_args.eval_repetitions,
                            instance.parallel_environments))
    for repetition in range(parsed_args.eval_repetitions):
        obs = env.reset()
        for episode in range(parsed_args.eval_episode_length):
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            print(action)
            obs, rewards, dones, info = env.step(action)
            rewards_all[episode, repetition] += rewards
            if np.all(dones):  # pragma: no cover
                break
    print(np.mean(rewards_all, (1, 2)))


def add_eval_parser(parser) -> None:
    """Fill in the parser with arguments for evaluating the model."""
    parser.add_argument("--eval-episode-length", default=5, type=int)
    parser.add_argument("--eval-repetitions", default=100, type=int)
    parser.set_defaults(func=eval_func)
