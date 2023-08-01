"""Policy of a random player."""
from random import choice
from typing import Any, Optional, cast

import numpy as np
import sb3_contrib.common.maskable.policies  # type: ignore


class RandomPolicy(sb3_contrib.common.maskable.policies.MaskableActorCriticPolicy):
    """Policy of a random player."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Do nothing."""

    def predict(self, *args: Any,
                action_masks: Optional[np.ndarray] = None, **kwargs: Any) -> tuple[np.ndarray, None]:
        """Get a random prediction."""
        if action_masks is None:
            raise AttributeError("Incorrect usage of random policy. Provide action masks.")
        number_of_envs = action_masks.shape[1]

        def get_choice(i):
            possible_indicies = [j for j in range(number_of_envs) if cast(np.ndarray, action_masks)[i, j]]
            return choice(possible_indicies)
        return np.fromiter((get_choice(i) for i in range(action_masks.shape[0])), int), None

    def to(self, *args: Any, **kwargs: Any) -> 'RandomPolicy':
        """Do nothing."""
        return self
