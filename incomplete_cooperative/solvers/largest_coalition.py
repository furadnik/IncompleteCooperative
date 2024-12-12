"""A greedy solver."""
from ..protocols import Gym
from ..run.model import ModelInstance


class LargestSolver:
    """Solve by selecting the largest explorable coalition."""

    def __init__(self, instance: ModelInstance | None = None) -> None:
        """Do nothing with the instance."""

    def next_step(self, gym: Gym) -> int:
        """Get the locally best next move."""
        valid_actions = [x for x in range(gym.action_masks().shape[0]) if gym.action_masks()[x]]
        valid_coalitions = [gym.explorable_coalitions[i] for i in valid_actions]
        max_coalition_size = max(map(len, valid_coalitions))
        best_actions = (act for act, coal in zip(valid_actions, valid_coalitions) if len(coal) == max_coalition_size)
        return next(best_actions)
