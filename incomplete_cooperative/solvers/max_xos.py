"""A MAX XOS solver."""

from ..protocols import Gym
from ..run.model import ModelInstance
from ..max_xos_approximation import compute_max_xos_approximation

import numpy as np


class MaxXosSolver:
    """Solve by Max XOS Approximation."""

    def __init__(self, instance: ModelInstance | None = None) -> None:
        """Initialize the variables."""
            
        self.max_xos = None
        self.num_of_queries = 0
        self.remaining_coalitions = []
        
    def after_reset(self, gym: Gym) -> None:
        icg_gym = gym.get_env()

        game = icg_gym.incomplete_game
        game._values[:, [game._values_lower_index, game._values_upper_index]] *= -1 # TODO: Do this in a clean way

        self.queried_coalition_ids, self.multiplicative_factor = compute_max_xos_approximation(game)

        self.num_of_queries = len(self.queried_coalition_ids) # This is stored so we can later learn when to stop plotting the graph

        # Compute the rest of the coalitions and shuffle them
        all_coalition_ids = np.arange(1, 2**game.number_of_players)
        filtered_coalition_ids = all_coalition_ids[~np.isin(all_coalition_ids, self.queried_coalition_ids)]
        self.remaining_coalition_ids = np.random.shuffle(filtered_coalition_ids)

    def next_step(self, gym: Gym) -> int:
        """Get the locally best next move."""
        if self.queried_coalition_ids.size > 0:
            popped_coalition = self.queried_coalition_ids[-1]
            self.queried_coalition_ids = self.queried_coalition_ids[:-1]
            return popped_coalition
        else:
            popped_coalition = self.remaining_coalitions[-1]
            self.remaining_coalition_ids = self.remaining_coalition_ids[:-1]
            return popped_coalition
