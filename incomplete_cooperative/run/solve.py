"""Solving script."""
import numpy as np

from incomplete_cooperative.solvers import SOLVERS

from .model import ModelInstance
from .save import Output, save


def solve_func(instance: ModelInstance, parsed_args) -> None:
    """Solve the game."""
    solver = SOLVERS[parsed_args.solver]()
    env = instance.env
    repetitions = parsed_args.solve_repetitions
    if instance.run_steps_limit is None:
        instance.run_steps_limit = 2**instance.number_of_players
    rewards_all = np.zeros((instance.run_steps_limit + 1,
                            repetitions))
    actions_all = np.zeros((instance.run_steps_limit,
                            repetitions))
    for repetition in range(repetitions):
        env.reset()
        rewards_all[0, repetition] = env.reward
        for episode in range(instance.run_steps_limit):
            action = solver.next_step(env)
            _, reward, done, _, info = env.step(action)
            rewards_all[episode + 1, repetition] = reward
            # map the `action` (index in explorable coalitions) to `coalition`.
            actions_all[episode, repetition] = info["chosen_coalition"]

            if done:  # pragma: no cover
                break

    exploitability = -rewards_all

    save(instance.model_dir, instance.unique_name,
         Output(exploitability, actions_all, parsed_args))


def add_solve_parser(parser) -> None:
    """Fill in the parser with arguments for solving the game."""
    parser.add_argument("--solve-repetitions", default=100, type=int)
    parser.add_argument("--solver", choices=SOLVERS.keys())
    parser.set_defaults(func=solve_func)
