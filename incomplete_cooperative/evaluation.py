"""Solving script."""
from typing import Callable

import numpy as np

from .protocols import GapFunction, Gym


def evaluate(get_next_step: Callable, env: Gym,
             repetitions: int, run_steps_limit: int, gap_func: GapFunction) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a game solver."""
    exploitability = np.zeros((run_steps_limit + 1, repetitions))
    actions_all = np.zeros((run_steps_limit, repetitions))
    for repetition in range(repetitions):
        _, base_info = env.reset()
        game = base_info["game"]
        incomplete_game = env.get_wrapper_attr("incomplete_game").copy()
        known_coalitions = env.get_wrapper_attr("initially_known_coalitions").copy()

        incomplete_game.set_known_values(game.get_values(known_coalitions), known_coalitions)
        incomplete_game.compute_bounds()
        exploitability[0, repetition] = gap_func(incomplete_game)
        for episode in range(run_steps_limit):
            action = get_next_step(env)
            _, _, done, _, info = env.step(action)
            known_coalitions.append(env.explorable_coalitions[action])

            # reveal the same coalitions on the non-normalized game, compute its exploitability
            incomplete_game.set_known_values(game.get_values(known_coalitions), known_coalitions)
            incomplete_game.compute_bounds()
            exploitability[episode + 1, repetition] = gap_func(incomplete_game)

            # map the `action` (index in explorable coalitions) to `coalition`.
            actions_all[episode, repetition] = info["chosen_coalition"]

            if done:  # pragma: no cover
                break

    return exploitability, actions_all
