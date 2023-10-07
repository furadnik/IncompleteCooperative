"""Custom solvers module."""
from incomplete_cooperative.protocols import Solver

from .greedy import GreedySolver
from .largest_coalition import LargestSolver
from .random import RandomSolver

SOLVERS: dict[str, type[Solver]] = {
    "greedy": GreedySolver,
    "random": RandomSolver,
    "largest": LargestSolver,
}
