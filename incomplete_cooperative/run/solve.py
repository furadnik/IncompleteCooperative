"""Solving script."""
from incomplete_cooperative.evaluation import evaluate
from incomplete_cooperative.solvers import SOLVERS

from .model import ModelInstance
from .save import Output, save


def solve_func(instance: ModelInstance, parsed_args) -> None:
    """Solve the game."""
    solver = SOLVERS[parsed_args.solver]()
    env = instance.env

    exploitability, actions_all = evaluate(
        solver.next_step, env, parsed_args.solve_repetitions,
        instance.run_steps_limit or 2**instance.number_of_players
    )

    save(instance.model_dir, instance.unique_name,
         Output(exploitability, actions_all, parsed_args))


def add_solve_parser(parser) -> None:
    """Fill in the parser with arguments for solving the game."""
    parser.add_argument("--solve-repetitions", default=100, type=int)
    parser.add_argument("--solver", choices=SOLVERS.keys())
    parser.set_defaults(func=solve_func)
