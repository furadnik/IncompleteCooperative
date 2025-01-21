"""Wrapper around `ICG_Gym` with linear size of observations and actions."""
from typing import Any

import gymnasium as gym  # type: ignore
import numpy as np
from numpy import ndarray

from .icg_gym import ICG_Gym
from .protocols import StepResult, Value


class ICG_Gym_Linear(gym.Env):

    def __init__(self, icg_gym: ICG_Gym) -> None:
        self.icg_gym = icg_gym

        self.N: int = icg_gym.incomplete_game.number_of_players
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.N, Value),
            high=np.ones(self.N, Value),
            dtype=Value)
        self.action_space = gym.spaces.Discrete(self.N)

        self.subset_sizes = np.array([bin(i.id).count("1") for i in icg_gym.explorable_coalitions])

    def _sum_values_of_the_same_size(self, x: ndarray) -> ndarray:
        """Sum the elements of exponential ndarray with corresponding coalition sizes."""
        assert x.shape == (len(self.icg_gym.explorable_coalitions), ), \
            f"Expected shape (2 ** {self.N} - {self.N} - 2,) = ({2 ** self.N - self.N - 2},), got {x.shape}"
        return np.bincount(self.subset_sizes, weights=x)

    def action_masks(self) -> np.ndarray:
        """An action is allowed if at least one Coalition of given size is unknown."""
        exponential_mask = self.icg_gym.action_masks()
        return self._sum_values_of_the_same_size(exponential_mask).astype(bool)

    def reset(self, **kwargs) -> tuple[ndarray, dict[str, Any]]:
        state, info = self.icg_gym.reset()
        linear_state = self._sum_values_of_the_same_size(state)
        return linear_state, info

    def step(self, coalition_size: int) -> StepResult:
        """Perform one step.

        The action is the size of the coalition to reveal.
        We sample the actual coalition of that size.
        """
        assert 0 <= coalition_size < self.N, f"Expected 0 <= coalition_size < {self.N}, got {coalition_size}"

        candidates = np.where((self.subset_sizes == coalition_size) * self.icg_gym.action_masks())[0]
        coalition_to_reveal = np.random.choice(candidates)

        exp_step_results = self.icg_gym.step(coalition_to_reveal)
        linear_state = self._sum_values_of_the_same_size(exp_step_results[0])
        return linear_state, *exp_step_results[1:]

    @property
    def done(self) -> bool:
        return self.icg_gym.done

    @property
    def reward(self) -> Value:
        return self.icg_gym.reward
