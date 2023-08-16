from abc import ABC, abstractmethod
from unittest import TestCase

from incomplete_cooperative.protocols import Solver
from incomplete_cooperative.solvers import GreedySolver

from .utils import GymMixin


class TestSolverMixin(GymMixin, ABC):
    """Test Solvers."""

    @abstractmethod
    def get_solver(self) -> Solver:
        """Get the solver."""

    def test_valid_run(self):
        """Test that the gym isn't affected at any point."""
        solver = self.get_solver()
        gym = self.get_gym(number_of_players=4)
        while not gym.done:
            prev_values = gym.incomplete_game._values.tolist()
            action = solver.next_step(gym)
            self.assertTrue(gym.incomplete_game._values.tolist() == prev_values)
            self.assertTrue(gym.valid_action_mask()[action])
            gym.step(action)


class TestGreedy(TestSolverMixin, TestCase):

    def get_solver(self) -> Solver:
        return GreedySolver()
