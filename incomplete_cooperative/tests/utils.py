"""Utils for testing."""
import numpy as np

from incomplete_cooperative.bounds import compute_bounds_superadditive
from incomplete_cooperative.coalitions import Coalition, all_coalitions
from incomplete_cooperative.exploitability import compute_exploitability
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.generators import factory_generator
from incomplete_cooperative.graph_game import GraphCooperativeGame
from incomplete_cooperative.icg_gym import ICG_Gym
from incomplete_cooperative.protocols import MutableIncompleteGame, Value


def trivial_fill(game: MutableIncompleteGame) -> None:
    """Trivially fill game."""
    for coalition in all_coalitions(game):
        game.set_value(len(coalition)**2, coalition)


class IncompleteGameMixin:
    """Mixin for testing incomplete game."""

    def get_game(self, *, number_of_players=6) -> IncompleteCooperativeGame:
        """Get game based on params."""
        return IncompleteCooperativeGame(number_of_players, compute_bounds_superadditive)

    def get_game_miss_coals(self, missed_coals=[Coalition(3)], filler=factory_generator,
                            number_of_players=6) -> IncompleteCooperativeGame:
        """Get game with pre-defined coalitions missing."""
        game = factory_generator(number_of_players, owner=0, bounds_computer=compute_bounds_superadditive)
        for coal in missed_coals:
            game.unset_value(coal)
        return game

    def get_game_minimal(self, number_of_players=6, **kwargs) -> IncompleteCooperativeGame:
        """Get game with minimal info."""
        return self.get_game_miss_coals(
            missed_coals=[c for c in all_coalitions(number_of_players) if len(c) not in [number_of_players, 1, 0]],
            number_of_players=number_of_players, **kwargs)


class GymMixin(IncompleteGameMixin):
    """Mixin for testing Gym."""

    def get_gym(self, *, known_coalitions=None, **kwargs) -> ICG_Gym:
        """Get gym based on params."""
        if known_coalitions is None:
            num_players = kwargs.get("number_of_players", 6)
            known_coalitions = [0, 2**num_players - 1] + [2**i for i in range(num_players)]
        incomplete_game = self.get_game(**kwargs)
        full_game = self.get_game(**kwargs)
        trivial_fill(full_game)
        transformed_known_coalitions = list(map(Coalition, filter(lambda x: x < 2**incomplete_game.number_of_players,
                                                                  known_coalitions)))
        return ICG_Gym(incomplete_game, lambda: full_game.copy(), transformed_known_coalitions, compute_exploitability)


class GraphGameMixin:
    """Mixin for testing graph game."""

    def get_game(self, n_players=4) -> GraphCooperativeGame:
        """Get the graph game."""
        return GraphCooperativeGame(np.random.rand(n_players, n_players).astype(Value))

    def to_incomplete(self, game: GraphCooperativeGame) -> IncompleteCooperativeGame:
        """Turn a graph game into an incomplete game."""
        incom = IncompleteCooperativeGame(game.number_of_players, compute_bounds_superadditive)
        incom.set_values(game.get_values(), all_coalitions(incom))
        return incom
