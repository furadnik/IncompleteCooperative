"""A random solver."""
from random import choice

from ..protocols import Gym


class RandomSolver:
    """Random solver implementation."""

    def next_step(self, gym: Gym) -> int:
        """Get the locally best next move."""
        valid_actions = [x for x in range(gym.valid_action_mask().shape[0]) if gym.valid_action_mask()[x]]
        return choice(valid_actions)
