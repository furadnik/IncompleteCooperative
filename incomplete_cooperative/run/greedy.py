"""Get best states from the coalitions."""
from functools import partial
from random import Random

import numpy as np

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.gameplay import (
    get_exploitabilities_of_action_sequence,
    get_stacked_exploitabilities_of_action_sequences)
from incomplete_cooperative.icg_gym import ICG_Gym
from incomplete_cooperative.protocols import GapFunction, Value

from .model import ModelInstance
from .save import Output, save

EPSILON = 1e-6


def greedy_func(instance: ModelInstance, parsed_args, randomize: bool = False) -> None:
    """Get greedy best states."""
    rng = Random(instance.seed) if randomize else None  # nosec
    if instance.run_steps_limit is None:  # pragma: no cover
        instance.run_steps_limit = 2**instance.number_of_players
    env = instance.get_env()
    assert isinstance(env, ICG_Gym)
    exploitability, best_coalitions = get_greedy_rewards(env, instance.run_steps_limit,
                                                         parsed_args.sampling_repetitions,
                                                         instance.gap_function_callable,
                                                         instance.parallel_environments,
                                                         rng)
    actions_all = np.reshape(np.array(best_coalitions), (len(best_coalitions), 1))
    save(instance.model_dir, instance.unique_name,
         Output(exploitability, actions_all, parsed_args))


def get_greedy_rewards(env: ICG_Gym, max_steps: int, repetitions: int, gap_func: GapFunction,
                       processes: int = 1, random: Random | None = None) -> tuple[np.ndarray, list[int]]:
    """Return best rewards.

    Returns: A tuple:
        A `np.ndarray` with the best rewards in each step.
        A list of lists of chosen coalitions at each step for the best results.
    """
    best_exploitabilities = -np.ones((max_steps + 1, repetitions))

    incomplete_game = env.incomplete_game
    generated_games = [env.generator() for _ in range(repetitions)]

    best_exploitabilities[0] = np.fromiter(get_exploitabilities_of_action_sequence(
        incomplete_game, generated_games, [], gap_func, processes=processes
    ), Value, count=repetitions)

    possible_actions = set(env.get_wrapper_attr("explorable_coalitions"))
    action_sequence: list[Coalition] = []
    possible_next_action_sequences: list[list[Coalition]] = [[]]
    while len(action_sequence) < max_steps:
        expected_exploitabilities = np.array(list(
            get_stacked_exploitabilities_of_action_sequences(
                incomplete_game, generated_games, possible_next_action_sequences,
                gap_func, processes=processes)))

        if random is not None:
            best_action_value = np.min(np.mean(expected_exploitabilities, axis=1))
            best_action_index = random.choice(  # nosec
                np.arange(expected_exploitabilities.shape[0])[np.mean(expected_exploitabilities, axis=1) - best_action_value < EPSILON]
            )
        else:
            best_action_index = int(np.argmin(np.mean(expected_exploitabilities, axis=1)))

        if possible_next_action_sequences[best_action_index]:
            best_action = possible_next_action_sequences[best_action_index][-1]
            action_sequence.append(best_action)
            possible_actions.remove(best_action)

        best_exploitabilities[len(action_sequence), :] = expected_exploitabilities[best_action_index]
        possible_next_action_sequences = [action_sequence + [action]
                                          for action in possible_actions]

    return best_exploitabilities, [x.id for x in action_sequence]


def add_greedy_parser(parser, randomize: bool = False) -> None:
    """Fill in the parser with arguments for solving the game."""
    parser.set_defaults(func=partial(greedy_func, randomize=randomize))
    parser.add_argument("--sampling-repetitions", default=1, type=int)
