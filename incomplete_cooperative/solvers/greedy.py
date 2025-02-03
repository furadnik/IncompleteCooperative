"""A greedy solver."""
from ..protocols import Gym, Value
from ..run.model import ModelInstance


class GreedySolver:
    """Greedy solver implementation."""

    def __init__(self, instance: ModelInstance | None = None, worst: bool = False) -> None:
        """Do nothing with the instance."""
        self.worst = worst

    def _next_action_value(self, gym: Gym, action: int) -> Value:
        """Return a value of the next step."""
        _, value, _, _, _ = gym.step(action)
        gym.unstep(action)
        return value

    def next_step(self, gym: Gym) -> int:
        """Get the locally best next move."""
        valid_actions = [x for x in range(gym.action_masks().shape[0]) if gym.action_masks()[x]]
        action_values = [self._next_action_value(gym, act) for act in valid_actions]
        max_action_value = max(action_values) if not self.worst else min(action_values)  # type: ignore[type-var]
        best_actions = (act for act, val in zip(valid_actions, action_values) if val == max_action_value)
        return next(best_actions)

    def after_reset(self, gym: Gym) -> None:
        """Do nothing."""
