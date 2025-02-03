"""A greedy solver."""

from ..protocols import Gym
from ..run.model import ModelInstance
from ..multiplicative import RootLinearApproximation

import numpy as np


class RLASolver:
    """Solve by Root Linear Approximation."""

    def __init__(self, instance: ModelInstance | None = None) -> None:
        """Initialize the variables."""
        
        self.rla = None
        self.num_of_queries = 0
        self.remaining_coalitions = []
    

    def on_reset(self, gym: Gym) -> None:
        icg_gym = instance.get_env()

        inverted_game = icg_gym.incomplete_game
        inverted_game.values = icg_gym.incomplete_game.values * -1 # The approximation algorithm needs submodular game

        self.rla = RootLinearApproximation(inverted_game)

        self.num_of_queries = len(self.rla.queried_coalitions) # This is stored so we can later learn when to stop plotting the graph

        # Compute the rest of the coalitions and shuffle them
        all_coalitions = np.arange(1, 2**icg_gym.incomplete_game.number_of_players)
        filtered_coalitions = all_coalitions[~np.isin(all_coalitions, self.rla.queried_coalitions)]
        self.remaining_coalitions = np.random.shuffle(filtered_coalitions)
        

    def next_step(self, gym: Gym) -> int:
        """Get the locally best next move."""
        if self.rla.queried_coalitions.size > 0:
            popped_coalition = self.rla.queried_coalitions[-1]
            self.rla.queried_coalitions = self.rla.queried_coalitions[:-1]
            return popped_coalition
        else:
            popped_coalition = self.remaining_coalitions[-1]
            self.remaining_coalitions = self.remaining_coalitions[:-1]
            return popped_coalition

