"""An Agent Gym for Incomplete Cooperative Games."""
from typing import Any, Callable, Iterable

import gymnasium as gym  # type: ignore
import numpy as np

from .coalitions import Coalition, all_coalitions, grand_coalition
from .normalize import NormalizableGame, normalize_game
from .protocols import (GapFunction, IncompleteGame, Info,
                        MutableIncompleteGame, State, StepResult, Value)


def compute_reward(game: IncompleteGame, gap_func: GapFunction) -> Value:
    """Compute the reward of the game."""
    return -gap_func(game)


class ICG_Gym(gym.Env):
    """A `gym.Env` for incomplete cooperative games."""

    def __init__(self, game: MutableIncompleteGame,
                 game_generator: Callable[[], NormalizableGame],
                 initially_known_coalitions: Iterable[Coalition],
                 gap_func: GapFunction,
                 done_after_n_actions: int | None = None) -> None:
        """Initialize gym.

        `game` is an object that has the necessary `bound_computer` functions.
        `full_game` is the game with all known values. We'll use it when we reveal new values.
        `normalized_game` is the same as full_game, but with values normalized using `normalize_game`.
        `initially_known_coalitions` are the codes of coalitions, whose values will be in the `game` from the start.
        """
        super().__init__()

        self.incomplete_game = game
        self.generator = game_generator

        self.gap_func = gap_func
        self.done_after_n_actions = done_after_n_actions
        self.steps_taken = 0

        self.initially_known_coalitions = list(set(initially_known_coalitions).union(
            {Coalition(0), grand_coalition(game)}))
        # explorable coalitions are those, whose values we initially do not know.
        self.explorable_coalitions = list(filter(lambda x: x not in self.initially_known_coalitions,
                                                 all_coalitions(self.full_game)))

        # setup the gym.
        self.reset()
        self.observation_space = gym.spaces.Box(
            low=np.zeros(len(self.explorable_coalitions), Value),
            high=np.ones(len(self.explorable_coalitions), Value) * self.full_game.get_value(
                grand_coalition(self.full_game)),
            dtype=Value)
        self.action_space = gym.spaces.Discrete(len(self.explorable_coalitions))

    def action_masks(self) -> np.ndarray:
        """Get valid actions for the agent."""
        return np.invert(self.incomplete_game.are_values_known(self.explorable_coalitions))

    @property
    def state(self) -> np.ndarray[Any, np.dtype[Value]]:
        """Get the current state."""
        normalized_values = self.normalized_game.get_values(self.explorable_coalitions)
        values_known = self.incomplete_game.are_values_known(self.explorable_coalitions)
        return normalized_values * values_known

    @property
    def reward(self) -> Value:  # type: ignore
        """Return reward -- negative exploitability."""
        return compute_reward(self.incomplete_game, self.gap_func)

    @property
    def done(self) -> bool:
        """Decide whether we are done -- all values are known."""
        return (
            self.done_after_n_actions is not None and self.steps_taken >= self.done_after_n_actions
        ) or (
            not np.any(self.action_masks())
        ) or bool(np.all(
            (self.incomplete_game.get_upper_bounds() - self.incomplete_game.get_lower_bounds()) == 0
        ))

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[State, Info]:
        """Reset the game into initial state."""
        super().reset(seed=seed, options=options)
        self.full_game = self.generator()
        self.normalized_game = self.full_game.copy()
        normalize_game(self.normalized_game)

        self.incomplete_game.set_known_values(self.full_game.get_values(self.initially_known_coalitions),
                                              self.initially_known_coalitions)
        self.incomplete_game.compute_bounds()
        self.steps_taken = 0

        return self.state, {"game": self.full_game}

    def step(self, action: int) -> StepResult:
        """Implement one step of the arbitor, reveal coalition and compute exploitability.

        Return the new state, reward, whether we're done, and some (empty) additional info.
        """
        # The chosen coalition for revealing, skipping the singletons
        chosen_coalition = self.explorable_coalitions[action]
        self.incomplete_game.reveal_value(self.full_game.get_value(chosen_coalition),
                                          chosen_coalition)
        self.incomplete_game.compute_bounds()
        self.steps_taken += 1

        return self.state, self.reward, self.done, False, {"chosen_coalition": chosen_coalition.id}

    def unstep(self, action: int) -> StepResult:
        """Undo a step of the arbitor.

        Return the new state, reward, whether we're done, and some (empty) additional info.
        """
        # The chosen coalition for revealing, skipping the singletons
        chosen_coalition = self.explorable_coalitions[action]
        self.incomplete_game.unreveal_value(chosen_coalition)
        self.incomplete_game.compute_bounds()
        self.steps_taken -= 1

        return self.state, self.reward, self.done, False, {"chosen_coalition": chosen_coalition.id}
