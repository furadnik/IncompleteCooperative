"""Tests for the `run` module."""
from unittest import TestCase

import numpy as np

from incomplete_cooperative.run.best_states import (best_states_func,
                                                    fill_in_coalitions)
from incomplete_cooperative.run.solve import solve_func

from .test_run_learn import GetLearningResultMixin


class TestBestStates(GetLearningResultMixin, TestCase):

    # account for float precision
    epsilon = 0.0000000001

    def test_first_step_same(self):
        args, instance = self.get_instance(number_of_players=4, solve_repetitions=1,
                                           run_steps_limit=6,
                                           solver="greedy", func="foobar")
        greedy_out = self.get_saver_output(solve_func, instance, args)
        best_out = self.get_saver_output(best_states_func, instance, args)
        self.assertEqual(greedy_out.avg_exploitabilities[0], best_out.avg_exploitabilities[0])
        self.assertEqual(greedy_out.avg_exploitabilities[1], best_out.avg_exploitabilities[1])
        for j in range(7):
            self.assertGreaterEqual(greedy_out.avg_exploitabilities[j] + self.epsilon,
                                    best_out.avg_exploitabilities[j])

    def test_fill_in_coalitions(self):
        coalitions = [
            [1],
            [0, 2],
            [0, 1, 2],
        ]
        expected = np.array([[1, np.nan, np.nan], [0, 2, np.nan], [0, 1, 2]])
        target_array = np.full((3, 3), np.nan)
        for i in range(len(coalitions)):
            fill_in_coalitions(target_array[i], coalitions[i])
        self.assertTrue(np.all(
            (target_array == expected) | (np.isnan(target_array) & np.isnan(expected))),
            msg=str(target_array) + str(expected))
