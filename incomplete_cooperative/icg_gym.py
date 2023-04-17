"""An Agent Gym for Incomplete Cooperative Games."""
from typing import Any, Iterable

import gym  # type: ignore
import numpy as np

from .coalitions import Coalition, all_coalitions
from .protocols import Game, MutableIncompleteGame, Value

State = np.ndarray
StepResult = tuple[np.ndarray[Any, np.dtype[Value]], Value, bool, Any]


class ICG_Gym(gym.Env):
    """A `gym.Env` for incomplete cooperative games."""

    def __init__(self, game: MutableIncompleteGame,
                 full_game: Game,
                 initially_known_coalitions: Iterable[Coalition]) -> None:
        """Initialize gym.

        `game` is an object that has the necessary `bound_computer` functions.
        `full_game` is the game with all known values. We'll use it when we reveal new values.
        `initially_known_coalitions` are the codes of coalitions, whose values will be in the `game` from the start.
        """
        super().__init__()
        self.initially_known_coalitions = list(set(initially_known_coalitions).union({Coalition(0)}))

        self.game = game
        self.full_game = full_game
        self.initially_known_values = list(full_game.get_values(self.initially_known_coalitions))

        # explorable coalitions are those, whose values we initially do not know.
        self.explorable_coalitions = list(set(filter(lambda x: x not in self.initially_known_coalitions,
                                                     all_coalitions(self.full_game))))

        # setup the gym.
        self.reset()
        self.observation_space = gym.spaces.Box(low=np.zeros(len(self.explorable_coalitions), np.float32),
                                                high=np.ones(len(self.explorable_coalitions), np.float32),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.explorable_coalitions))

    def valid_action_mask(self) -> np.ndarray:
        """Get valid actions for the agent."""
        return np.invert(self.game.are_values_known(self.explorable_coalitions))

    @property
    def state(self) -> np.ndarray[Any, np.dtype[Value]]:
        """Get the current state."""
        return np.nan_to_num(self.game.get_known_values(self.explorable_coalitions))

    @property
    def reward(self) -> Value:  # type: ignore
        """Return reward -- negative exploitability."""  # TODO: implement later.

    @property
    def done(self) -> bool:
        """Decide whether we are done -- all values are known."""
        return bool(np.all(self.game.are_values_known()))

    def reset(self) -> State:
        """Reset the game into initial state."""
        self.game.set_known_values(self.initially_known_values,
                                   self.initially_known_coalitions)
        self.game.compute_bounds()

        return self.state

    def step(self, action: int) -> StepResult:
        """Implement one step of the arbitor, reveal coalition and compute exploitability.

        Return the new state, reward, whether we're done, and some (empty) additional info.
        """
        # The chosen coalition for revealing, skipping the singletons
        chosen_coalition = self.explorable_coalitions[action]
        self.game.reveal_value(self.full_game.get_value(chosen_coalition),
                               chosen_coalition)

        return self.state, self.reward, self.done, {}
