"""Get best states from the coalitions."""
from typing import Iterator

import numpy as np

from incomplete_cooperative.icg_gym import ICG_Gym
from incomplete_cooperative.protocols import Value

from .model import ModelInstance
from .save import Output, save


def best_states_func(instance: ModelInstance, parsed_args) -> None:
    """Get best states."""
    env = instance.env
    if instance.run_steps_limit is None:  # pragma: no cover
        instance.run_steps_limit = 2**instance.number_of_players
    rewards_all = np.zeros((instance.run_steps_limit + 1,
                            1))
    actions_all = np.zeros((instance.run_steps_limit,
                            1))
    rewards_all[0, 0] = env.reward
    best_rewards = get_best_rewards(instance.env, instance.run_steps_limit)
    for episode in range(instance.run_steps_limit):  # TODO: implement later.
        rewards_all[episode + 1, 0] = best_rewards[episode]

    exploitability = -rewards_all
    save(instance.model_dir, instance.unique_name,
         Output(exploitability, actions_all, parsed_args))


def get_best_rewards(env: ICG_Gym, max_steps: int) -> np.ndarray:
    """Return best rewards."""
    initial_reward = env.reward
    best_rewards = np.full((max_steps,), initial_reward)
    for steps, reward in get_values(env, max_steps):
        if steps != 0:
            best_rewards[steps - 1] = max(best_rewards[steps - 1], reward)
    return best_rewards


def get_values(env: ICG_Gym, max_steps: int,
               steps_taken: int = 0, current_index: int = 0) -> Iterator[tuple[int, Value]]:
    """Get values for steps."""
    if steps_taken >= max_steps:
        yield steps_taken, env.reward
        return
    if current_index >= env.valid_action_mask().shape[0]:
        yield steps_taken, env.reward
        return
    yield from get_values(env, max_steps, steps_taken, current_index + 1)
    env.step(current_index)
    yield from get_values(env, max_steps, steps_taken + 1, current_index + 1)
    env.unstep(current_index)


def add_best_states_parser(parser) -> None:
    """Fill in the parser with arguments for solving the game."""
    parser.set_defaults(func=best_states_func)
