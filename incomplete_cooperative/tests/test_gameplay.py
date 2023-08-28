from unittest import TestCase

import numpy as np

from incomplete_cooperative.coalitions import (Coalition, all_coalitions,
                                               minimal_game_coalitions)
from incomplete_cooperative.exploitability import compute_exploitability
from incomplete_cooperative.gameplay import (
    apply_action_sequence, get_exploitabilities_of_action_sequences,
    possible_action_sequences, possible_next_actions)

from .utils import IncompleteGameMixin


class TestPossibleNextActions(IncompleteGameMixin, TestCase):

    def test_actions_are_missing_coalitions(self):
        cases = [
            [],
            [Coalition(1)],
            [Coalition(1), Coalition(2), Coalition(3)],
            list(all_coalitions(self.get_game()))
        ]
        for case in cases:
            game = self.get_game_miss_coals(case)
            self.assertEqual(list(possible_next_actions(game)), case)


class TestPossActionSequences(IncompleteGameMixin, TestCase):
    cases = [
        ([], [[]], 10),
        ([Coalition(1)], [[], [Coalition(1)]], None),
        ([Coalition(1), Coalition(2), Coalition(3)],
         [[], [Coalition(1)], [Coalition(2)], [Coalition(3)]], 1),
        (list(all_coalitions(6)), [[]] + [[Coalition(i)] for i in range(2**6)], 1)
    ]

    def test_ac_seqs_are_subsets_of_missing(self):
        for miss, expected, max_len in self.cases:
            game = self.get_game_miss_coals(miss)
            self.assertEqual(list(possible_action_sequences(game, max_size=max_len)),
                             expected)

    def test_all_unique(self):
        game = self.get_game_miss_coals(all_coalitions(3))
        ac_sequences = list(possible_action_sequences(game))
        for sequence in ac_sequences:
            self.assertEqual(len([x for x in ac_sequences if x == sequence]), 1)

    def test_all_sequences(self):
        game = self.get_game_miss_coals(all_coalitions(4), number_of_players=4)
        self.assertEqual(len(list(possible_action_sequences(game))), 2**(2**4))

    def test_max_size_correct_values(self):
        game = self.get_game_miss_coals(all_coalitions(3))
        ac_sequences = list(possible_action_sequences(game))
        for sequence in possible_action_sequences(game, 9):
            self.assertLessEqual(len(sequence), 9)
            self.assertIn(sequence, ac_sequences)

    def test_max_size_not_missing_any(self):
        game = self.get_game_miss_coals(all_coalitions(3))
        ac_sequences = list(possible_action_sequences(game, 9))
        for sequence in possible_action_sequences(game):
            if len(sequence) <= 9:
                self.assertIn(sequence, ac_sequences)


class TestApplySequence(IncompleteGameMixin, TestCase):

    def test_apply_nothing(self):
        full = self.get_game_miss_coals([])
        apply_action_sequence(full, self.get_game_miss_coals([]), [])
        for coalition in all_coalitions(full.number_of_players):
            if len(coalition):
                self.assertFalse(full.is_value_known(coalition))

    def test_apply_all(self):
        full = self.get_game_miss_coals(all_coalitions(6))
        apply_action_sequence(full, self.get_game_miss_coals([]), list(all_coalitions(6)))
        self.assertTrue(np.all(full.get_values() == self.get_game_miss_coals([]).get_values()))

    def test_apply_nonexisting(self):
        self.assertRaises(IndexError, apply_action_sequence,
                          self.get_game_miss_coals([]), self.get_game_miss_coals([]), [Coalition(1000000)])

    def test_extra_include(self):
        cases = [
            # include, reveal
            ([Coalition(x) for x in range(10)], []),
            ([Coalition(x) for x in range(10)], [Coalition(x) for x in range(10, 20)]),
            ([Coalition(x) for x in range(10)], [Coalition(x) for x in range(5, 15)]),
            ([], [Coalition(x) for x in range(10)]),
        ]
        for include, reveal in cases:
            with self.subTest(include=include, reveal=reveal):
                game = self.get_game_miss_coals(all_coalitions(6))
                game_full = self.get_game_miss_coals([])
                apply_action_sequence(game, game_full, reveal, include=include)
                for coalition in all_coalitions(6):
                    if coalition in include or coalition in reveal:
                        self.assertEqual(game.get_value(coalition), game_full.get_value(coalition))
                    else:
                        self.assertFalse(game.is_value_known(coalition))


class TestActionSeqExploitabilities(IncompleteGameMixin, TestCase):

    def test_get_actions(self):
        cases_missed_coal = [
            [Coalition(x) for x in range(1, 10) if len(Coalition(x)) > 1],  # singletons must be known to compute bounds
            [Coalition(x) for x in range(7, 10) if len(Coalition(x)) > 1],
            [],
        ]
        for include in cases_missed_coal:
            with self.subTest(include=include):
                game = self.get_game_miss_coals(include, number_of_players=4)
                game_full = self.get_game_miss_coals([], number_of_players=4)
                for action_sequence, _ in get_exploitabilities_of_action_sequences(
                        game, game_full):
                    self.assertEqual(set(action_sequence).union(include), set(include))

    def actions_exploitability_correct(self) -> None:
        missed_coalitions = [x for x in all_coalitions(4) if x not in minimal_game_coalitions(
            self.get_game(number_of_players=4))]
        for action_sequence, exploitability in get_exploitabilities_of_action_sequences(
            game=self.get_game_miss_coals(missed_coalitions, number_of_players=4),
            full_game=self.get_game(number_of_players=4)
        ):
            new_game = self.get_game_miss_coals(number_of_players=4)
            for action in action_sequence:
                new_game.reveal_value(self.get_game(number_of_players=4).get_value(action), action)
            self.assertEqual(compute_exploitability(new_game), exploitability)
