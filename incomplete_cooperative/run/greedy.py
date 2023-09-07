"""Get best states from the coalitions."""
from typing import Iterator, cast

import numpy as np

from incomplete_cooperative.coalitions import Coalition, all_coalitions
from incomplete_cooperative.gameplay import \
    get_exploitabilities_of_action_sequences
from incomplete_cooperative.icg_gym import ICG_Gym
from incomplete_cooperative.protocols import Value

from .model import ModelInstance
from .save import Output, save


def greedy_func(instance: ModelInstance, parsed_args) -> None:
    """Get best states."""
    env = instance.env
    if instance.run_steps_limit is None:  # pragma: no cover
        instance.run_steps_limit = 2**instance.number_of_players
    rewards_all = np.zeros((instance.run_steps_limit + 1,
                            1))
    actions_all = np.full((instance.run_steps_limit,
                           instance.run_steps_limit), np.nan)
    rewards_all[0, 0] = env.reward
    greedy_rewards, greedy_coalitions = get_greedy_rewards(instance.env, instance.run_steps_limit,
                                                           parsed_args.sampling_repetitions)
    for episode in range(instance.run_steps_limit):
        rewards_all[episode + 1, 0] = greedy_rewards[episode]
        fill_in_coalitions(actions_all[episode], greedy_coalitions[episode])
    exploitability = -rewards_all
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
    assert not np.any(env.incomplete_game.are_values_known(env.explorable_coalitions))  # nosec
    assert np.all(env.incomplete_game.are_values_known(env.initially_known_coalitions))  # nosec
    greedy_rewards = np.full((max_steps,), initial_reward)
    greedy_actions: list[list[int]] = [[] for _ in range(max_steps)]
    action_value_map = list((sorted(a, key=lambda x: x.id), v) for a, v in get_values(env, repetitions, max_steps))

    print(action_value_map)
    print(max_steps)

    def get_action_value(action_sequence) -> Value | None:
        action_sequence = sorted(action_sequence, key=lambda x: x.id)
        return next((v for a, v in action_value_map if a == action_sequence), None)

    possible_actions = list(all_coalitions(env.full_game))
    action_sequence: list[Coalition] = []
    while len(action_sequence) < max_steps:
        next_action_sequences = (action_sequence + [a] for a in possible_actions)
        next_sequences_w_rewards = list(filter(lambda x: x[1] is not None,
                                        ((a, get_action_value(a)) for a in next_action_sequences)))
        action_sequence, value = max(next_sequences_w_rewards, key=lambda x: cast(Value, x[1]))
        assert value is not None  # nosec
        steps = len(action_sequence)
        coalitions = [x.id for x in action_sequence]
        reward = -value
        print(action_sequence)

        if steps != 0 and greedy_rewards[steps - 1] < reward:
            greedy_rewards[steps - 1] = reward
            greedy_actions[steps - 1] = coalitions

    return greedy_rewards, greedy_actions


def get_values(env: ICG_Gym, repetitions: int, max_steps: int) -> Iterator[tuple[list[Coalition], Value]]:
    """Get by sampling."""
    initial = list(get_exploitabilities_of_action_sequences(env.incomplete_game, env.full_game, max_steps))
    actions = (x[0] for x in initial)
    values = np.fromiter((x[1] for x in initial), Value)
    for _ in range(repetitions - 1):  # pragma: no cover
        env.reset()
        values += np.fromiter((x[1] for x in get_exploitabilities_of_action_sequences(
            env.incomplete_game, env.full_game, max_steps)),
            Value, values.shape[0])
    values /= repetitions
    return zip(actions, values)


def add_greedy_parser(parser) -> None:
    """Fill in the parser with arguments for solving the game."""
    parser.set_defaults(func=greedy_func)
    parser.add_argument("--sampling-repetitions", default=1, type=int)
