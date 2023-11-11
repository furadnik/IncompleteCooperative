"""A random solver."""
from random import Random

from ..protocols import Gym
from ..run.model import ModelInstance


class RandomSolver:
    """Random solver implementation."""

    def __init__(self, instance: ModelInstance) -> None:
        """Save generator."""
        self._generator = Random(3 * instance.seed) if instance is not None else Random()

    def next_step(self, gym: Gym) -> int:
        """Get the locally best next move."""
        valid_actions = [x for x in range(gym.valid_action_mask().shape[0]) if gym.valid_action_mask()[x]]
        if not valid_actions:  # pragma: no cover
            return 0
        return self._generator.choice(valid_actions)  # nosec - random not used as a security feature
