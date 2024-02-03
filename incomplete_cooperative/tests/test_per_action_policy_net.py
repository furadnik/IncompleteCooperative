from unittest import TestCase

import gymnasium as gym
import numpy as np
import torch  # type: ignore

from incomplete_cooperative.per_action_policy_net import (
    CustomPerActionPolicyNet, get_action_encoding)
from incomplete_cooperative.protocols import Value

from .utils import GymMixin

MAX_PLAYERS = 9
MIN_PLAYERS = 3


class TestGetEncoding(TestCase, GymMixin):

    def test_no_zero_lines(self):
        for i in range(MIN_PLAYERS, MAX_PLAYERS):
            encoding = get_action_encoding(i)
            summed = encoding.sum(1)
            self.assertTrue(torch.all(summed != 0))

    def test_last_almost_full(self):
        for i in range(MIN_PLAYERS, MAX_PLAYERS):
            encoding = get_action_encoding(i)
            summed = encoding[-1].sum()
            self.assertEqual(summed, i - 1)

    def test_concrete_thing(self):
        encoding = get_action_encoding(3)
        expected = torch.tensor([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ])
        self.assertTrue(torch.all(encoding == expected), encoding)

    def test_matches_gym(self):
        for p in range(MIN_PLAYERS, MAX_PLAYERS):
            encoding = get_action_encoding(p)
            gym = self.get_gym(number_of_players=p)
            for i, coalition in enumerate(gym.explorable_coalitions):
                array = torch.zeros(p)
                array[list(coalition.players)] = 1
                self.assertTrue(torch.all(array == encoding[i]))


def get_box(players: int) -> gym.spaces.Box:
    """Get a sample box environment."""
    max_actions = 2**players - players - 2
    return gym.spaces.Box(
        low=np.zeros(max_actions, Value),
        high=np.ones(max_actions, Value),
        dtype=Value)


class TestPerActionPolNet(TestCase):

    def test_encoding_players(self):
        for p in range(MIN_PLAYERS, MAX_PLAYERS):
            net = CustomPerActionPolicyNet(get_box(p), number_of_players=p)
            self.assertTrue(torch.all(net.encoding == get_action_encoding(p)))

    def test_insufficient_pls(self):
        self.assertRaises(ValueError, CustomPerActionPolicyNet, get_box(3), 1)

    def test_forward(self):
        for p in range(MIN_PLAYERS, MAX_PLAYERS):
            net = CustomPerActionPolicyNet(get_box(p), number_of_players=p)
            observations = torch.zeros(2**p - p - 2)
            net.forward(observations)
