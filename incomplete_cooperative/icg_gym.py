"""An Agent Gym for Incomplete Cooperative Games."""
import gym
import numpy as np
from .game import IncompleteCooperativeGame, CoalitionPlayers

State = np.ndarray
StepResult = tuple  # TODO: type hints better.


class ICG_Gym(gym.Env):
    """A `gym.Env` for incomplete cooperative games."""

    def __init__(self, game: IncompleteCooperativeGame,
                 full_game: IncompleteCooperativeGame,
                 initially_known_values: list[CoalitionPlayers]) -> None:
        """Initialize gym."""
        super().__init__()
        self.game = game
        self.full_game = full_game
        self.initially_known_values = {players: full_game.get_value(full_game.players_to_coalition(players))
                                       for players in initially_known_values}

        # explorable coalitions are those, whose values we initially do not know.
        initially_known_coalitions = set(map(self.full_game.players_to_coalition, initially_known_values))
        self.explorable_coalitions = list(filter(lambda x: x not in initially_known_coalitions,
                                                 self.full_game.coalitions))

        self.reset()
        self.observation_space = gym.spaces.Box(low=np.zeros(len(self.explorable_coalitions)),
                                                high=np.ones(len(self.explorable_coalitions)),
                                                dtype=np.float32)

        self.action_space = gym.spaces.Discrete(len(self.explorable_coalitions))

    def valid_action_mask(self) -> np.ndarray:
        """Get valid actions for the agent."""
        return self.game.known_values

    @property
    def state(self) -> np.ndarray:
        """Get the current state."""
        return self.full_game.get_values * self.game.known_values

    @property
    def reward(self) -> int:
        """Return reward -- negative exploitability."""
        return -self.game.exploitability  # TODO: implement later.

    def reset(self) -> State:
        """Reset the game into initial state."""
        self.game.set_known_values(self.initially_known_values)
        self.game.compute_bounds()
        # TODO: normalize the game.

        return self.state

    def step(self, action: int) -> StepResult:
        """Implement one step of the arbitor, reveal coalition and compute exploitability.

        Return the new state, reward, whether we're done, and some (empty) additional info.
        """
        # The chosen coalition for revealing, skipping the singletons
        chosen_coalition = self.powerset[action + self.num_players]
        self.nodes[chosen_coalition][0] = 1
        self.game.compute_bounds()

        return self.state, self.reward, self.done, {}
