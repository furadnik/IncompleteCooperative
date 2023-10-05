"""Get best states from the coalitions."""
import numpy as np

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.gameplay import (
    get_stacked_exploitabilities_of_action_sequences,
    sample_exploitabilities_of_action_sequences)
from incomplete_cooperative.icg_gym import ICG_Gym
from incomplete_cooperative.protocols import Value

from .model import ModelInstance
from .save import Output, save


def best_states_func(instance: ModelInstance, parsed_args) -> None:
    """Get best states."""
    if instance.run_steps_limit is None:  # pragma: no cover
        instance.run_steps_limit = 2**instance.number_of_players
    actions_all = np.full((instance.run_steps_limit + 1,
                           parsed_args.eval_repetitions,
                           instance.run_steps_limit), np.nan)
    exploitability = None
    best_coalitions: list[list[list[int]]] = []
    for repetition in range(parsed_args.eval_repetitions):
        exploitability_rep, new_best_coalitions = get_best_exploitability(instance.env, instance.run_steps_limit,
                                                                          parsed_args.sampling_repetitions,
                                                                          processes=instance.parallel_environments)
        if exploitability is None:
            exploitability = exploitability_rep
        else:
            exploitability = np.hstack((exploitability, exploitability_rep))

        if not best_coalitions:
            best_coalitions = [[x] for x in new_best_coalitions]
        else:
            best_coalitions = [x + [y] for x, y in zip(best_coalitions, new_best_coalitions)]
    assert exploitability is not None  # nosec

    for episode in range(len(best_coalitions)):
        fill_in_coalitions(actions_all[episode], best_coalitions[episode])
    save(instance.model_dir, instance.unique_name,
         Output(exploitability, actions_all, parsed_args))


def fill_in_coalitions(target_array: np.ndarray, coalitions: list[list[int]]) -> None:
    """Fill in the coalitions one by one to the target array.

    Note: `target_array` is an output parameter.
    """
    for i, coal in enumerate(coalitions):
        for j, c in enumerate(coal):
            target_array[i, j] = c


def get_best_exploitability(env: ICG_Gym, max_steps: int, repetitions: int, processes: int = 1
                            ) -> tuple[np.ndarray, list[list[int]]]:
    """Return best rewards.

    Returns: A tuple:
        A `np.ndarray` with the best rewards in each step.
        A list of lists of chosen coalitions at each step for the best results.
    """
    initial_placeholder_value = -1
    sum_best_exploitabilities = np.full((max_steps + 1, 1), initial_placeholder_value, dtype=Value)
    best_actions: list[list[int]] = [[] for _ in range(max_steps + 1)]

    games = [env.generator() for _ in range(repetitions)]
    sum_game = games[0]
    for g in games[1:]:
        sum_game += g  # type: ignore[operator]

    sample_actions, sample_values = sample_exploitabilities_of_action_sequences(
        env.incomplete_game, lambda x: sum_game, samples=1, max_size=max_steps, processes=processes)
    for i, act_sequence in enumerate(sample_actions):
        steps = len(act_sequence)
        coalitions = [x.id for x in act_sequence]

        if np.mean(sum_best_exploitabilities[steps]) == initial_placeholder_value or \
                np.mean(sum_best_exploitabilities[steps]) > np.mean(sample_values[:, i]):
            sum_best_exploitabilities[steps] = sample_values[:, i]
            best_actions[steps] = coalitions

    best_exploitabilities = np.fromiter(
        get_stacked_exploitabilities_of_action_sequences(
            env.incomplete_game, games,
            ([Coalition(c) for c in x] for x in best_actions),  # turn best actions to coalitions again for a second
            processes=processes),
        count=len(best_actions), dtype=Value)

    return best_exploitabilities, best_actions


def add_best_states_parser(parser) -> None:
    """Fill in the parser with arguments for solving the game."""
    parser.set_defaults(func=best_states_func)
    parser.add_argument("--sampling-repetitions", default=1, type=int)
    parser.add_argument("--eval-repetitions", default=1, type=int)
