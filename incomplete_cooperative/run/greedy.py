"""Get best states from the coalitions."""

import numpy as np

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.gameplay import \
    get_stacked_exploitabilities_of_action_sequences
from incomplete_cooperative.icg_gym import ICG_Gym

from .model import ModelInstance
from .save import Output, save


def greedy_func(instance: ModelInstance, parsed_args) -> None:
    """Get greedy best states."""
    if instance.run_steps_limit is None:  # pragma: no cover
        instance.run_steps_limit = 2**instance.number_of_players
    exploitability, best_coalitions = get_greedy_rewards(instance.env, instance.run_steps_limit,
                                                         parsed_args.sampling_repetitions,
                                                         instance.parallel_environments)
    actions_all = np.reshape(np.array(best_coalitions), (len(best_coalitions), 1))
    save(instance.model_dir, instance.unique_name,
         Output(exploitability, actions_all, parsed_args))


def get_greedy_rewards(env: ICG_Gym, max_steps: int, repetitions: int, processes: int = 1
                       ) -> tuple[np.ndarray, list[int]]:
    """Return best rewards.

    Returns: A tuple:
        A `np.ndarray` with the best rewards in each step.
        A list of lists of chosen coalitions at each step for the best results.
    """
    initial_exploitability = env.reward
    best_exploitabilities = np.full((max_steps + 1, repetitions), -initial_exploitability)

    incomplete_game = env.incomplete_game
    generated_games = [env.generator() for _ in range(repetitions)]

    possible_actions = set(env.explorable_coalitions)
    action_sequence: list[Coalition] = []
    possible_next_action_sequences: list[list[Coalition]] = [[]]
    while len(action_sequence) < max_steps:
        expected_exploitabilities = np.array(list(
            get_stacked_exploitabilities_of_action_sequences(
                incomplete_game, generated_games, possible_next_action_sequences, processes=processes)))

        best_action_index = np.argmin(np.mean(expected_exploitabilities, axis=1))

        if possible_next_action_sequences[best_action_index]:
            best_action = possible_next_action_sequences[best_action_index][-1]
            action_sequence.append(best_action)
            possible_actions.remove(best_action)

        best_exploitabilities[len(action_sequence), :] = expected_exploitabilities[best_action_index]
        possible_next_action_sequences = [action_sequence + [action]
                                          for action in possible_actions]

    return best_exploitabilities, [x.id for x in action_sequence]


def add_greedy_parser(parser) -> None:
    """Fill in the parser with arguments for solving the game."""
    parser.set_defaults(func=greedy_func)
    parser.add_argument("--sampling-repetitions", default=1, type=int)
