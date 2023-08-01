import random
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from incomplete_cooperative.random_player import RandomPolicy


def get_mask(number: int, unavailable: list[int], layers: int = 1) -> np.ndarray:
    """Get an action mask."""
    mask = np.ones((layers, number), bool)
    for f in unavailable:
        mask[:, f] = 0

    return mask


class TestRandomPlayer(TestCase):

    tests: list[tuple[int, list[int], int, str]] = [
        (100, [], 1, "One layer."),
        (100, [], 10, "Multiple layers."),
        (10, [0, 1, 2, 4, 5, 6, 7, 8, 9], 10, "Disallowed."),
    ]

    def assertValidMask(self, players, disallowed, layers):
        mask = get_mask(players, disallowed, layers)
        policy = self.get_random_policy()
        for _ in range(100):
            result, _ = policy.predict(action_masks=mask)
            for i in result:
                self.assertNotIn(i, disallowed)
                self.assertIn(i, range(players))

    def get_random_policy(self):
        return RandomPolicy()

    def test_all_actions(self):
        for players, disallowed, layers, name in self.tests:
            with self.subTest(name=name):
                self.assertValidMask(players, disallowed, layers)

    def test_no_action_mask(self):
        self.assertRaises(AttributeError, self.get_random_policy().predict)
