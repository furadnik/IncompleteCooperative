from unittest import TestCase

import numpy as np

from incomplete_cooperative.coalitions import Coalition
from incomplete_cooperative.regret import (GameRegretMinimizer,
                                           metacoalition_ids_by_coalition_size)


class TestGetCoalitionsIdMap(TestCase):

    def test_simple_example(self):
        number_of_players = 3
        limit_of_revealed = 2
        expected = [0, 1, 2, 4, 3, 5, 6]
        self.assertEqual(list(metacoalition_ids_by_coalition_size(number_of_players, limit_of_revealed)),
                         expected)

    def test_example_limit_equal_possible(self):
        number_of_players = 3
        limit_of_revealed = 3
        expected = [0, 1, 2, 4, 3, 5, 6, 7]
        self.assertEqual(list(metacoalition_ids_by_coalition_size(number_of_players, limit_of_revealed)),
                         expected)

    def test_example_limit_greater_than_possible(self):
        number_of_players = 3
        limit_of_revealed = 60
        expected = [0, 1, 2, 4, 3, 5, 6, 7]
        self.assertEqual(list(metacoalition_ids_by_coalition_size(number_of_players, limit_of_revealed)),
                         expected)

    def test_supersets_are_supersets(self):
        number_of_players = 5
        previous_result = metacoalition_ids_by_coalition_size(number_of_players, 1)
        for limit_of_revealed in range(2, 7):
            current_result = metacoalition_ids_by_coalition_size(number_of_players, limit_of_revealed)
            self.assertTrue(
                np.all(
                    current_result[:previous_result.size] == previous_result
                )
            )
            previous_result = current_result


class TestGameRegretMinimizer(TestCase):

    def test_metacoalition_ids(self):
        minimizer = GameRegretMinimizer(3, 2)
        self.assertEqual(minimizer.get_metacoalition_id([Coalition(3)]), 1,
                         msg=minimizer.coalitions_to_player_ids)

    def test_metacoalition_ids_non_viable(self):
        minimizer = GameRegretMinimizer(3, 2)
        self.assertEqual(minimizer.get_metacoalition_id([Coalition(2), Coalition(3)]), 1,
                         msg=minimizer.coalitions_to_player_ids)

    def test_metacoalition_ids_multiple(self):
        minimizer = GameRegretMinimizer(3, 2)
        self.assertEqual(minimizer.get_metacoalition_id([Coalition(3), Coalition(5)]), 3,
                         msg=minimizer.coalitions_to_player_ids)

    def test_apply_regret(self):
        minimizer = GameRegretMinimizer(3, 2)
        print(minimizer.coalitions_to_player_ids, minimizer.meta_id_to_rank)
        minimizer.regret_min_iteration(np.array([1, 0, 0]), [
            [Coalition(3), Coalition(5)],
            [Coalition(5), Coalition(6)],
            [Coalition(3), Coalition(6)],
        ])
        self.assertTrue(np.allclose(minimizer.cumulative_regret[0], [1 / 6, 1 / 6, -2 / 6]),
                        msg=minimizer.cumulative_regret)
        self.assertTrue(np.allclose(minimizer.get_average_strategy([]),
                                    [0, 0, 0, 1 / 3, 0, 1 / 3, 1 / 3, 0]),
                        msg=minimizer.get_average_strategy([]))

    def test_apply_regret_twice(self):
        minimizer = GameRegretMinimizer(3, 2)
        print(minimizer.coalitions_to_player_ids, minimizer.meta_id_to_rank)
        minimizer.regret_min_iteration(np.array([1, 0, 0]), [
            [Coalition(3), Coalition(5)],
            [Coalition(5), Coalition(6)],
            [Coalition(3), Coalition(6)],
        ])
        minimizer.regret_min_iteration(np.array([0, 1, 0]), [
            [Coalition(3), Coalition(5)],
            [Coalition(5), Coalition(6)],
            [Coalition(3), Coalition(6)],
        ])
        self.assertTrue(np.allclose(minimizer.cumulative_regret[0], [1 / 6, 1 / 6, 1 / 6]),
                        msg=minimizer.cumulative_regret)
        self.assertTrue(np.allclose(minimizer.get_average_strategy([]),
                                    [0, 0, 0, 5 / 12, 0, 5 / 12, 1 / 6, 0]),
                        msg=minimizer.get_average_strategy([]))

    def test_apply_regret_plus(self):
        minimizer = GameRegretMinimizer(3, 2, plus=True)
        print(minimizer.coalitions_to_player_ids, minimizer.meta_id_to_rank)
        minimizer.regret_min_iteration(np.array([1, 0, 0]), [
            [Coalition(3), Coalition(5)],
            [Coalition(5), Coalition(6)],
            [Coalition(3), Coalition(6)],
        ])
        self.assertTrue(np.allclose(minimizer.cumulative_regret[0], [1 / 6, 1 / 6, 0]),
                        msg=minimizer.cumulative_regret)
        self.assertTrue(np.allclose(minimizer.get_average_strategy([]),
                                    [0, 0, 0, 1 / 3, 0, 1 / 3, 1 / 3, 0]),
                        msg=minimizer.get_average_strategy([]))

    def test_apply_regret_plus_twice(self):
        minimizer = GameRegretMinimizer(3, 2, plus=True)
        print(minimizer.coalitions_to_player_ids, minimizer.meta_id_to_rank)
        minimizer.regret_min_iteration(np.array([1, 0, 0]), [
            [Coalition(3), Coalition(5)],
            [Coalition(5), Coalition(6)],
            [Coalition(3), Coalition(6)],
        ])
        minimizer.regret_min_iteration(np.array([0, 1, 0]), [
            [Coalition(3), Coalition(5)],
            [Coalition(5), Coalition(6)],
            [Coalition(3), Coalition(6)],
        ])
        self.assertTrue(np.allclose(minimizer.cumulative_regret[0], [1 / 6, 1 / 6, 1 / 2]),
                        msg=minimizer.cumulative_regret)
        self.assertTrue(np.allclose(minimizer.get_average_strategy([]),
                                    [0, 0, 0, 4 / 9, 0, 4 / 9, 1 / 9, 0]),
                        msg=minimizer.get_average_strategy([]))

    def test_initial_strat_uniform(self):
        minimizer = GameRegretMinimizer(3, 2, plus=True)
        self.assertTrue(np.allclose(minimizer.get_average_strategy([]), [0, 0, 0, 1 / 3, 0, 1 / 3, 1 / 3, 0]),
                        msg=minimizer.cumulative_regret)
        self.assertTrue(np.allclose(minimizer.get_average_strategy([Coalition(3)]), [0, 0, 0, 0, 0, 1 / 2, 1 / 2, 0]),
                        msg=minimizer.cumulative_regret)
