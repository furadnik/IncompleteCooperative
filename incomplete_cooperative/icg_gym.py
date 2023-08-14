"""An Agent Gym for Incomplete Cooperative Games."""
from typing import Any, Callable, Iterable

import gymnasium as gym  # type: ignore
import numpy as np

from .coalitions import Coalition, all_coalitions, grand_coalition
from .exploitability import compute_exploitability
from .protocols import Game, MutableIncompleteGame, Value

State = np.ndarray
Info = dict[str, Any]
StepResult = tuple[np.ndarray[Any, np.dtype[Value]], Value, bool, bool, Info]


class ICG_Gym(gym.Env):
    """A `gym.Env` for incomplete cooperative games."""

    def __init__(self, game: MutableIncompleteGame,
                 game_generator: Callable[[], Game],
                 initially_known_coalitions: Iterable[Coalition],
                 done_after_n_actions: int | None = None) -> None:
        """Initialize gym.

        `game` is an object that has the necessary `bound_computer` functions.
        `full_game` is the game with all known values. We'll use it when we reveal new values.
        `initially_known_coalitions` are the codes of coalitions, whose values will be in the `game` from the start.
        """
        super().__init__()

        self.incomplete_game = game
        self.generator = game_generator
        self.full_game = game_generator()

        self.done_after_n_actions = done_after_n_actions
        self.steps_taken = 0

        self.initially_known_coalitions = list(set(initially_known_coalitions).union({Coalition(0)}))
        # explorable coalitions are those, whose values we initially do not know.
        self.explorable_coalitions = list(set(filter(lambda x: x not in self.initially_known_coalitions,
                                                     all_coalitions(self.full_game))))

        # setup the gym.
        self.reset()
        self.observation_space = gym.spaces.Box(
            low=np.zeros(len(self.explorable_coalitions), Value),
            high=np.ones(len(self.explorable_coalitions), Value) * self.full_game.get_value(
                grand_coalition(self.full_game)),
            dtype=Value)
        self.action_space = gym.spaces.Discrete(len(self.explorable_coalitions))

    def valid_action_mask(self) -> np.ndarray:
        """Get valid actions for the agent."""
        return np.invert(self.incomplete_game.are_values_known(self.explorable_coalitions))

    @property
    def state(self) -> np.ndarray[Any, np.dtype[Value]]:
        """Get the current state."""
        return np.nan_to_num(self.incomplete_game.get_known_values(self.explorable_coalitions))

    @property
    def reward(self) -> Value:  # type: ignore
        """Return reward -- negative exploitability."""
        return -compute_exploitability(self.incomplete_game)

    @property
    def done(self) -> bool:
        """Decide whether we are done -- all values are known."""
        if self.done_after_n_actions is None:
            return bool(np.all(
                (self.incomplete_game.get_upper_bounds() - self.incomplete_game.get_lower_bounds()) == 0))
        return self.steps_taken >= self.done_after_n_actions

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[State, Info]:
        """Reset the game into initial state."""
        super().reset(seed=seed, options=options)
        self.full_game = self.generator()
        self.incomplete_game.set_known_values(self.full_game.get_values(self.initially_known_coalitions),
                                              self.initially_known_coalitions)
        self.incomplete_game.compute_bounds()
        self.steps_taken = 0

        return self.state, {}

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
