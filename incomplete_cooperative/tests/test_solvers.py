from abc import ABC, abstractmethod
from unittest import TestCase

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.protocols import Solver
from incomplete_cooperative.run.model import ModelInstance
from incomplete_cooperative.solvers import (SOLVERS, GreedySolver,
                                            LargestSolver, RandomSolver)

from .utils import GymMixin


class TestSolverMixin(GymMixin, ABC):
    """Test Solvers."""

    @abstractmethod
    def get_solver(self) -> Solver:
        """Get the solver."""

    def test_solver_in_solvers(self):
        solver = self.get_solver()
        self.assertTrue(any(isinstance(x, type) and isinstance(solver, x) for x in SOLVERS.values()))

    def test_valid_run(self):
        """Test that the gym isn't affected at any point."""
        solver = self.get_solver()
        gym = self.get_gym(number_of_players=4)
        solver.after_reset(gym)
        while not gym.done:
            prev_values = gym.incomplete_game._values.tolist()
            action = solver.next_step(gym)
            self.assertTrue(gym.incomplete_game._values.tolist() == prev_values)
            self.assertTrue(gym.action_masks()[action])
            gym.step(action)


class TestGreedy(TestSolverMixin, TestCase):

    def get_solver(self) -> Solver:
        return GreedySolver()


class TestWorstGreedy(TestSolverMixin, TestCase):

    def get_solver(self) -> Solver:
        return GreedySolver(worst=True)


class TestRandom(TestSolverMixin, TestCase):

    def get_solver(self) -> Solver:
        return RandomSolver(ModelInstance())


class TestLargest(TestSolverMixin, TestCase):

    def get_solver(self) -> Solver:
        return LargestSolver()

    def test_largest_coalition(self):
        solver = self.get_solver()
        for i in range(3, 7):
            with self.subTest(number_of_players=i):
                gym = self.get_gym(number_of_players=i)
                solver.after_reset(gym)
                last_coalition_size = i
                while not gym.done:
                    action = solver.next_step(gym)
                    coalition_size = len(
                        Coalition(gym.step(action)[-1]['chosen_coalition'])  # turn id into Coalition
                    )
                    self.assertLessEqual(coalition_size, last_coalition_size)
                    last_coalition_size = coalition_size
