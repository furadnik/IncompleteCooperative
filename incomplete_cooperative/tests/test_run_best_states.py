"""Tests for the `run` module."""
from unittest import TestCase

import numpy as np

from incomplete_cooperative.run.best_states import (best_states_func,
                                                    fill_in_coalitions)
from incomplete_cooperative.run.solve import solve_func

from .test_run_learn import GetLearningResultMixin


class TestBestStates(GetLearningResultMixin, TestCase):

    # account for float precision
    epsilon = 0.000001

    def test_first_step_same(self):
        args, instance = self.get_instance(number_of_players=4, solve_repetitions=1,
                                           run_steps_limit=6, sampling_repetitions=1,
                                           eval_repetitions=1,
                                           solver="greedy", func="foobar", game_generator="factory_fixed")
        greedy_out = self.get_saver_output(solve_func, instance, args)
        best_out = self.get_saver_output(best_states_func, instance, args)
        self.assertEqual(greedy_out.avg_data[0], best_out.avg_data[0], greedy_out.avg_data)
        self.assertEqual(greedy_out.avg_data[1], best_out.avg_data[1], greedy_out.avg_data)
        for j in range(7):
            self.assertGreaterEqual(greedy_out.avg_data[j] + self.epsilon,
                                    best_out.avg_data[j], j)

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
        self.assertTrue(np.array_equal(target_array, expected, equal_nan=True),
                        msg=str(target_array) + str(expected))

    def test_factory_fixed_always_same(self):
        args, instance = self.get_instance(number_of_players=4, solve_repetitions=1,
                                           sampling_repetitions=1,
                                           eval_repetitions=1,
                                           run_steps_limit=5, func="foobar", game_generator="factory_fixed")
        reference = self.get_saver_output(best_states_func, instance, args)
        for j in range(1, 7):
            args, instance = self.get_instance(number_of_players=4, sampling_repetitions=j,
                                               eval_repetitions=1,
                                               run_steps_limit=5, func="foobar", game_generator="factory_fixed")
            output = self.get_saver_output(best_states_func, instance, args)
            for x, y in zip(reference.avg_data, output.avg_data):
                self.assertAlmostEqual(x, y)

    def test_data_shape_multiple_eval_rep(self):
        args, instance = self.get_instance(number_of_players=4, solve_repetitions=1,
                                           run_steps_limit=6, sampling_repetitions=6,
                                           eval_repetitions=3,
                                           solver="best_states", func="foobar",
                                           game_generator="factory_fixed")
        best_out = self.get_saver_output(best_states_func, instance, args)
        self.assertEqual(best_out.data.shape, (7, 3 * 6))
        self.assertAlmostEqual(np.all(np.abs(best_out.avg_data - best_out.data[:, 0])), 0)
