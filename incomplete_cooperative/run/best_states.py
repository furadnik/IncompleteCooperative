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
    actions_all = np.full((instance.run_steps_limit,
                           instance.run_steps_limit), np.nan)
    rewards_all[0, 0] = env.reward
    best_rewards, best_coalitions = get_best_rewards(instance.env, instance.run_steps_limit)
    for episode in range(instance.run_steps_limit):
        rewards_all[episode + 1, 0] = best_rewards[episode]
        fill_in_coalitions(actions_all[episode], best_coalitions[episode])
    exploitability = -rewards_all
    save(instance.model_dir, instance.unique_name,
         Output(exploitability, actions_all, parsed_args))


def fill_in_coalitions(target_array: np.ndarray, coalitions: list[int]) -> None:
    """Fill in the coalitions one by one to the target array.

    Note: `target_array` is an output parameter.
    """
    for i, coal in enumerate(coalitions):
        target_array[i] = coal


def get_best_rewards(env: ICG_Gym, max_steps: int) -> tuple[np.ndarray, list[list[int]]]:
    """Return best rewards.

    Returns: A tuple:
        A `np.ndarray` with the best rewards in each step.
        A list of lists of chosen coalitions at each step for the best results.
    """
    initial_reward = env.reward
    best_rewards = np.full((max_steps,), initial_reward)
    best_actions: list[list[int]] = [[] for _ in range(max_steps)]
    for steps, coalitions, reward in get_values(env, max_steps):
        if steps != 0 and best_rewards[steps - 1] < reward:
            best_rewards[steps - 1] = reward
            best_actions[steps - 1] = coalitions
    return best_rewards, best_actions


def get_values(env: ICG_Gym, max_steps: int, chosen_coalitions: list[int] = [],
               steps_taken: int = 0, current_index: int = 0) -> Iterator[tuple[int, list[int], Value]]:
    """Get values for steps."""
    if steps_taken >= max_steps:
        yield steps_taken, list(chosen_coalitions), env.reward
        return
    if current_index >= env.valid_action_mask().shape[0]:
        yield steps_taken, list(chosen_coalitions), env.reward
        return
    yield from get_values(env, max_steps, chosen_coalitions, steps_taken, current_index + 1)
    _, _, _, _, info = env.step(current_index)
    yield from get_values(env, max_steps, chosen_coalitions + [info["chosen_coalition"]],
                          steps_taken + 1, current_index + 1)
    env.unstep(current_index)


def add_best_states_parser(parser) -> None:
    """Fill in the parser with arguments for solving the game."""
    parser.set_defaults(func=best_states_func)
