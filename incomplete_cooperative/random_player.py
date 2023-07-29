"""Policy of a random player."""
from random import choice
from typing import Any, Optional, Self

import numpy as np
import sb3_contrib.common.maskable.policies  # type: ignore


class RandomPolicy(sb3_contrib.common.maskable.policies.MaskableActorCriticPolicy):
    """Policy of a random player."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Dummy init."""

    def predict(self, *args: Any,
                action_masks: Optional[np.ndarray] = None, **kwargs: Any) -> tuple[np.ndarray, None]:
        """Get a random prediction."""
        if action_masks is None:
            raise AttributeError("Incorrect usage of random policy. Provide action masks.")
        possible_indicies = [i for i in range(action_masks.shape[1]) if np.all(action_masks[:, i])]
        return np.fromiter([choice(possible_indicies)], int), None

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Do nothing."""
        return self
