"""Solving script."""
from copy import deepcopy
from multiprocessing import Pool

import numpy as np

from .protocols import GapFunction, Gym, NextStep


def evaluate(get_next_step: NextStep, env: Gym,
             repetitions: int, run_steps_limit: int, gap_func: GapFunction,
             processes: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a game solver."""
    exploitability = np.zeros((run_steps_limit + 1, repetitions))
    actions = np.zeros((run_steps_limit, repetitions))

    def new_env() -> Gym:
        """Get a new gym environment, with a new incomplete game."""
        env.reset()
        return deepcopy(env)

    with Pool(processes=processes) as p:
        for repetition, (step_exploitability, step_actions) in enumerate(
            p.starmap(evaluate_episode,
                      ((get_next_step, new_env(), gap_func, run_steps_limit) for _ in range(repetitions)))
        ):
            exploitability[:, repetition] = step_exploitability
            actions[:, repetition] = step_actions

    return exploitability, actions


def evaluate_episode(next_step: NextStep, env: Gym,
                     gap_func: GapFunction, run_steps_limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one repetition of the evaluation.

    Arguments:
        next_step: The solver that computes the next step action.
        run_steps_limit: The maximum number of steps.
        incomplete_game: Incomplete game to play with. Will be modified.
        game: The full game from which to copy in the values. Won't be modified.

    Returns:
        A tuple of ndarrays:
            Exploitability (run_steps_limit+1,): The exploitabilities along the way (including initial).
            Actions (run_steps_limit,): The actions chosen at any given time.
    """
    exploitability = np.zeros((run_steps_limit + 1,))
    actions = np.zeros((run_steps_limit,))

    # get a copy of the game, which is not normalized
    _, base_info = env.reset()
    game = base_info["game"]
    incomplete_game = env.incomplete_game.copy()
    known_coalitions = env.initially_known_coalitions.copy()

    # initialize the copy of the incomplete game, which copies the gym, but isn't normalized
    incomplete_game.set_known_values(game.get_values(known_coalitions), known_coalitions)
    incomplete_game.compute_bounds()

    exploitability[0] = gap_func(incomplete_game)
    for episode in range(run_steps_limit):
        action = next_step(env)
        _, _, done, _, info = env.step(action)
        known_coalitions.append(env.explorable_coalitions[action])

        # reveal the same coalitions on the non-normalized game, compute its exploitability
        incomplete_game.set_known_values(game.get_values(known_coalitions), known_coalitions)
        incomplete_game.compute_bounds()
        exploitability[episode + 1] = gap_func(incomplete_game)

        # map the `action` (index in explorable coalitions) to `coalition`.
        actions[episode] = info["chosen_coalition"]

        if done:  # pragma: no cover
            break

    return exploitability, actions
