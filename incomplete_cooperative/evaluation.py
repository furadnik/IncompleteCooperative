"""Solving script."""
from itertools import starmap
from multiprocessing import Pool
from typing import Callable

import numpy as np

from .protocols import GapFunction, Gym, GymGenerator


def evaluate(get_next_step: Callable, env_generator: GymGenerator,
             repetitions: int, run_steps_limit: int, gap_func: GapFunction,
             processes: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a game solver."""
    call_arg_sequence = ((get_next_step, env_generator(), run_steps_limit, gap_func) for _ in range(repetitions))
    if processes > 1:
        with Pool(processes=processes) as p:  # TODO: seed
            exploitabilities_and_actions = p.starmap(eval_one, call_arg_sequence)
    else:  # do not use multiprocessing if we only have one process
        exploitabilities_and_actions = list(starmap(eval_one, call_arg_sequence))
    exploitability = np.vstack([x[0] for x in exploitabilities_and_actions]).T
    actions_all = np.vstack([x[1] for x in exploitabilities_and_actions]).T
    assert exploitability.shape == (run_steps_limit + 1, repetitions)
    assert actions_all.shape == (run_steps_limit, repetitions)
    return exploitability, actions_all


def eval_one(get_next_step: Callable, env: Gym, run_steps_limit: int, gap_func: GapFunction
             ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one environment."""
    exploitability = np.zeros(run_steps_limit + 1)
    actions_all = np.zeros(run_steps_limit)
    env.reset()

    exploitability[0] = env.reward
    for episode in range(run_steps_limit):
        action = get_next_step(env)
        _, reward, done, _, info = env.step(action)
        chosen_coalition = info["chosen_coalition"]
        exploitability[episode + 1] = reward

        # map the `action` (index in explorable coalitions) to `coalition`.
        actions_all[episode] = chosen_coalition

        if done:  # pragma: no cover
            break
    return exploitability, actions_all
