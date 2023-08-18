"""Evaluating script."""
import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks  # type: ignore

from .model import ModelInstance
from .save import Output, save


def eval_func(instance: ModelInstance, parsed_args) -> None:
    """Evaluate the model."""
    model = instance.model
    env = instance.env_generator()
    if instance.run_steps_limit is None:
        instance.run_steps_limit = 2**instance.number_of_players
    rewards_all = np.zeros((instance.run_steps_limit + 1,
                            parsed_args.eval_repetitions, instance.parallel_environments))
    actions_all = np.zeros((instance.run_steps_limit,
                            parsed_args.eval_repetitions, instance.parallel_environments))
    for repetition in range(parsed_args.eval_repetitions):
        obs = env.reset()
        assert isinstance(obs, np.ndarray)  # nosec
        rewards_all[0, repetition, :] = env.get_attr("reward")
        for episode in range(instance.run_steps_limit):
            action_masks = get_action_masks(env)
            action, _ = model.predict(
                obs, action_masks=action_masks, deterministic=parsed_args.eval_deterministic)
            obs, rewards, dones, info = env.step(action)
            assert isinstance(obs, np.ndarray)
            rewards_all[episode + 1, repetition, :] += rewards
            # map the `action` (index in explorable coalitions) to `coalition`.
            actions_all[episode, repetition, :] = np.vectorize(lambda x: x["chosen_coalition"])(info)

            if np.all(dones):  # pragma: no cover
                break

    exploitability = -rewards_all.reshape(
        instance.run_steps_limit + 1,
        parsed_args.eval_repetitions * instance.parallel_environments)
    actions_compact = actions_all.reshape(
        instance.run_steps_limit,
        parsed_args.eval_repetitions * instance.parallel_environments)

    save(instance.model_dir, instance.unique_name,
         Output(exploitability, actions_compact, parsed_args))


def add_eval_parser(parser) -> None:
    """Fill in the parser with arguments for evaluating the model."""
    parser.add_argument("--eval-output-path", default=".", type=str)
    parser.add_argument("--eval-repetitions", default=3000, type=int)
    parser.add_argument("--eval-deterministic", action="store_true")
    parser.set_defaults(func=eval_func)
