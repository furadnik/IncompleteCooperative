"""Custom solvers module."""
from functools import partial

from incomplete_cooperative.protocols import Solver

from .greedy import GreedySolver
from .largest_coalition import LargestSolver
from .random import RandomSolver

SOLVERS: dict[str, type[Solver]] = {
    "greedy": GreedySolver,
    "greedy_worst": partial(GreedySolver, worst=True),  # type: ignore[dict-item]
    "random": RandomSolver,
    "largest": LargestSolver,
}
