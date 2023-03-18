"""An Agent Gym for Incomplete Cooperative Games."""
from typing import Any, Iterable, cast

import gym  # type: ignore
import numpy as np

from .game import Coalition, IncompleteCooperativeGame, Value

State = np.ndarray
StepResult = tuple[np.ndarray[Any, np.dtype[Value]], Value, bool, Any]


class ICG_Gym(gym.Env):
    """A `gym.Env` for incomplete cooperative games."""

    def __init__(self, game: IncompleteCooperativeGame,
                 full_game: IncompleteCooperativeGame,
                 initially_known_coalitions: Iterable[Coalition]) -> None:
        """Initialize gym.

        `game` is an object that has the necessary `bound_computer` functions.
        `full_game` is the game with all known values. We'll use it when we reveal new values.
        `initially_known_coalitions` are the codes of coalitions, whose values will be in the `game` from the start.
        """
        super().__init__()
        initially_known_coalitions = list(set(initially_known_coalitions).union({0}))

        self.game = game
        # TODO: normalize the game.
        self.full_game = full_game
        self.initially_known_values = {coalition: cast(Value, full_game.get_value(coalition))
                                       for coalition in initially_known_coalitions}

        # explorable coalitions are those, whose values we initially do not know.
        self.explorable_coalitions = list(set(filter(lambda x: x not in initially_known_coalitions,
                                                     self.full_game.coalitions)))

        # setup the gym.
        self.reset()
        self.observation_space = gym.spaces.Box(low=np.zeros(len(self.explorable_coalitions), np.float32),
                                                high=np.ones(len(self.explorable_coalitions), np.float32),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.explorable_coalitions))

    @property
    def valid_action_mask(self) -> np.ndarray:
        """Get valid actions for the agent."""
        return np.fromiter(map(
            lambda coalition: 0 if self.game.is_value_known(coalition) else 1,
            self.explorable_coalitions
        ), bool)

    @property
    def state(self) -> np.ndarray[Any, np.dtype[Value]]:
        """Get the current state."""
        return self.full_game.values * self.game.known_values

    @property
    def reward(self) -> Value:  # type: ignore
        """Return reward -- negative exploitability."""  # TODO: implement later.

    @property
    def done(self) -> bool:
        """Decide whether we are done -- all values are known."""
        return self.game.full

    def reset(self) -> State:
        """Reset the game into initial state."""
        self.game.set_known_values(self.initially_known_values)
        self.game.compute_bounds()

        return self.state

    def step(self, action: int) -> StepResult:
        """Implement one step of the arbitor, reveal coalition and compute exploitability.

        Return the new state, reward, whether we're done, and some (empty) additional info.
        """
        # The chosen coalition for revealing, skipping the singletons
        chosen_coalition = self.explorable_coalitions[action]
        self.game.reveal_value(chosen_coalition,
                               cast(Value, self.full_game.get_value(chosen_coalition)))
        self.game.compute_bounds()

        return self.state, self.reward, self.done, {}
