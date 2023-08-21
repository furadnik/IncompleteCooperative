"""Tests for the `run` module."""
from unittest import TestCase

from incomplete_cooperative.run.best_states import best_states_func
from incomplete_cooperative.run.solve import solve_func

from .test_run_learn import GetLearningResultMixin


class TestBestStates(GetLearningResultMixin, TestCase):

    def test_first_step_same(self):
        args, instance = self.get_instance(number_of_players=4, solve_repetitions=1,
                                           run_steps_limit=6,
                                           solver="greedy", func="foobar")
        greedy_out = self.get_saver_output(solve_func, instance, args)
        best_out = self.get_saver_output(best_states_func, instance, args)
        self.assertEqual(greedy_out.avg_exploitability[0], best_out.avg_exploitability[0])
        self.assertEqual(greedy_out.avg_exploitability[1], best_out.avg_exploitability[1])
        for j in range(7):
            self.assertGreaterEqual(greedy_out.avg_exploitability[j],
                                    best_out.avg_exploitability[j])
