"""Evaluating script."""
import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks  # type: ignore

from .model import ModelInstance
from .save import save


def eval_func(instance: ModelInstance, parsed_args) -> None:
    """Evaluate the model."""
    model = instance.model
    print(parsed_args)
    env = instance.env_generator()
    rewards_all = np.zeros((parsed_args.eval_episode_length + 1,
                            parsed_args.eval_repetitions, instance.parallel_environments))
    actions_all = np.zeros((parsed_args.eval_episode_length,
                            parsed_args.eval_repetitions, instance.parallel_environments))
    for repetition in range(parsed_args.eval_repetitions):
        obs = env.reset()
        rewards_all[0, repetition, :] = env.get_attr("reward")
        for episode in range(parsed_args.eval_episode_length):
            action_masks = get_action_masks(env)
            action, _ = model.predict(
                obs, action_masks=action_masks, deterministic=parsed_args.eval_deterministic)
            obs, rewards, dones, info = env.step(action)
            rewards_all[episode + 1, repetition, :] += rewards

            if np.all(dones):  # pragma: no cover
                break

    exploitability = -rewards_all.reshape(
        parsed_args.eval_episode_length + 1,
        parsed_args.eval_repetitions * instance.parallel_environments)
    actions_compact = actions_all.reshape(
        parsed_args.eval_episode_length,
        parsed_args.eval_repetitions * instance.parallel_environments)

    save(exploitability, actions_compact, instance.model_out_path, parsed_args)


def add_eval_parser(parser) -> None:
    """Fill in the parser with arguments for evaluating the model."""
    parser.add_argument("--eval-output-path", default=".", type=str)
    parser.add_argument("--eval-episode-length", default=5, type=int)
    parser.add_argument("--eval-repetitions", default=100, type=int)
    parser.add_argument("--eval-deterministic", action="store_true")
    parser.set_defaults(func=eval_func)
