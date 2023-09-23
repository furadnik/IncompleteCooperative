"""Get best states from the coalitions."""
from typing import cast

import numpy as np

from incomplete_cooperative.coalitions import Coalition, all_coalitions
from incomplete_cooperative.gameplay import \
    sample_exploitabilities_of_action_sequences
from incomplete_cooperative.icg_gym import ICG_Gym

from .model import ModelInstance
from .save import Output, save


def greedy_func(instance: ModelInstance, parsed_args) -> None:
    """Get greedy best states."""
    if instance.run_steps_limit is None:  # pragma: no cover
        instance.run_steps_limit = 2**instance.number_of_players
    actions_all = np.full((instance.run_steps_limit + 1,
                           instance.run_steps_limit), np.nan)
    exploitability, best_coalitions = get_greedy_rewards(instance.env, instance.run_steps_limit,
                                                         parsed_args.sampling_repetitions)
    for episode in range(len(best_coalitions)):
        fill_in_coalitions(actions_all[episode], best_coalitions[episode])
    save(instance.model_dir, instance.unique_name,
         Output(exploitability, actions_all, parsed_args))


def fill_in_coalitions(target_array: np.ndarray, coalitions: list[int]) -> None:
    """Fill in the coalitions one by one to the target array.

    Note: `target_array` is an output parameter.
    """
    for i, coal in enumerate(coalitions):
        target_array[i] = coal


def get_greedy_rewards(env: ICG_Gym, max_steps: int, repetitions: int) -> tuple[np.ndarray, list[list[int]]]:
    """Return best rewards.

    Returns: A tuple:
        A `np.ndarray` with the best rewards in each step.
        A list of lists of chosen coalitions at each step for the best results.
    """
    initial_reward = env.reward
    best_exploitabilities = np.full((max_steps + 1, repetitions), -initial_reward)
    greedy_actions: list[list[int]] = [[] for _ in range(max_steps + 1)]
    sample_actions, sample_values = sample_exploitabilities_of_action_sequences(
        env.incomplete_game, lambda x: env.generator(), repetitions, max_size=max_steps)
    action_value_map = list((sorted(a, key=lambda x: x.id), i) for i, a in enumerate(sample_actions))

    print(action_value_map)
    print(max_steps)

    def get_action_index(action_sequence) -> int | None:
        action_sequence = sorted(action_sequence, key=lambda x: x.id)
        return next((i for a, i in action_value_map if a == action_sequence), None)

    possible_actions = list(all_coalitions(env.full_game))
    action_sequence: tuple[list[Coalition], int] = [], 0
    while len(action_sequence[0]) < max_steps:
        next_action_sequences = (action_sequence[0] + [a] for a in possible_actions)
        next_action_indices = list(filter(lambda x: x[1] is not None,
                                          ((a, get_action_index(a)) for a in next_action_sequences)))
        # get the optimal action sequence
        action_sequence = cast(tuple[list[Coalition], int],
                               max(next_action_indices, key=lambda x: np.mean(sample_values[:, cast(int, x[1])])))
        steps = len(action_sequence[0])
        coalitions = [x.id for x in action_sequence[0]]

        if np.mean(best_exploitabilities[steps]) > np.mean(sample_values[:, action_sequence[1]]):
            best_exploitabilities[steps] = sample_values[:, action_sequence[1]]
            greedy_actions[steps] = coalitions

    return best_exploitabilities, greedy_actions


def add_greedy_parser(parser) -> None:
    """Fill in the parser with arguments for solving the game."""
    parser.set_defaults(func=greedy_func)
    parser.add_argument("--sampling-repetitions", default=1, type=int)
