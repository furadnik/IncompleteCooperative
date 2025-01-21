"""Wrapper around `ICG_Gym` with linear size of observations and actions."""
import gymnasium as gym  # type: ignore
import numpy as np

from .icg_gym import ICG_Gym
from .protocols import Info, State, StepResult, Value


class ICG_Gym_Linear(gym.Env):

    def __init__(self, icg_gym: ICG_Gym) -> None:
        self.icg_gym = icg_gym

        self.number_of_players = icg_gym.incomplete_game.number_of_players
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.number_of_players, Value),
            high=np.ones(self.number_of_players, Value),
            dtype=Value)
        self.action_space = gym.spaces.Discrete(self.number_of_players)

        self.subset_sizes = np.array([len(coalition) for coalition in icg_gym.explorable_coalitions])

    def _sum_values_of_the_same_size(self, x: np.ndarray) -> np.ndarray:
        """Sum the elements of exponential ndarray with corresponding coalition sizes."""
        assert x.shape == (len(self.icg_gym.explorable_coalitions), ), \
            f"Expected shape (2 ** {self.number_of_players} - {self.number_of_players} - 2,) = ({len(self.icg_gym.explorable_coalitions)},), got {x.shape}"
        return np.bincount(self.subset_sizes, weights=x)

    def action_masks(self) -> np.ndarray:
        """Allow an action if at least one Coalition of given size is unknown."""
        exponential_mask = self.icg_gym.action_masks()
        return self._sum_values_of_the_same_size(exponential_mask).astype(bool)

    def reset(self, **kwargs) -> tuple[State, Info]:
        """Reset the environment."""
        state, info = self.icg_gym.reset()
        linear_state = self._sum_values_of_the_same_size(state)
        return linear_state, info

    @property
    def state(self) -> np.ndarray:
        """Fetch the state of the underlying gym and make it linear."""
        return self._sum_values_of_the_same_size(self.icg_gym.state)

    def step(self, coalition_size: int) -> StepResult:
        """Perform one step.

        The action is the size of the coalition to reveal.
        We sample the actual coalition of that size.
        """
        assert 0 <= coalition_size < self.number_of_players, f"Expected 0 <= coalition_size < {self.number_of_players}, got {coalition_size}"

        candidates = np.where((self.subset_sizes == coalition_size) * self.icg_gym.action_masks())[0]
        coalition_to_reveal = np.random.choice(candidates)

        exp_step_results = self.icg_gym.step(coalition_to_reveal)
        linear_state = self._sum_values_of_the_same_size(exp_step_results[0])
        return linear_state, *exp_step_results[1:]

    @property
    def done(self) -> bool:
        """Return whether or not we are done."""
        return self.icg_gym.done

    @property
    def reward(self) -> Value:
        """Return reward in this state."""
        return self.icg_gym.reward
