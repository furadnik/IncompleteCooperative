"""Get best states from the coalitions."""
from typing import Iterator

import numpy as np

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.gameplay import \
    get_exploitabilities_of_action_sequences
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
    best_rewards, best_coalitions = get_best_rewards(instance.env, instance.run_steps_limit,
                                                     parsed_args.sampling_repetitions)
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


def get_best_rewards(env: ICG_Gym, max_steps: int, repetitions: int) -> tuple[np.ndarray, list[list[int]]]:
    """Return best rewards.

    Returns: A tuple:
        A `np.ndarray` with the best rewards in each step.
        A list of lists of chosen coalitions at each step for the best results.
    """
    initial_reward = env.reward
    assert not np.any(env.incomplete_game.are_values_known(env.explorable_coalitions))  # nosec
    assert np.all(env.incomplete_game.are_values_known(env.initially_known_coalitions))  # nosec
    best_rewards = np.full((max_steps,), initial_reward)
    best_actions: list[list[int]] = [[] for _ in range(max_steps)]
    for act_sequence, value in get_values(env, repetitions, max_steps):
        steps = len(act_sequence)
        coalitions = [x.id for x in act_sequence]
        reward = -value

        if steps != 0 and best_rewards[steps - 1] < reward:
            best_rewards[steps - 1] = reward
            best_actions[steps - 1] = coalitions
    return best_rewards, best_actions


def get_values(env: ICG_Gym, repetitions: int, max_steps: int) -> Iterator[tuple[list[Coalition], Value]]:
    """Get by sampling."""
    initial = list(get_exploitabilities_of_action_sequences(env.incomplete_game, env.full_game, max_steps))
    actions = (x[0] for x in initial)
    values = np.fromiter((x[1] for x in initial), Value)
    for _ in range(repetitions - 1):
        env.reset()
        values += np.fromiter((x[1] for x in get_exploitabilities_of_action_sequences(
            env.incomplete_game, env.full_game, max_steps)),
            Value, values.shape[0])
    values /= repetitions
    return zip(actions, values)


def add_best_states_parser(parser) -> None:
    """Fill in the parser with arguments for solving the game."""
    parser.set_defaults(func=best_states_func)
    parser.add_argument("--sampling-repetitions", default=1, type=int)
