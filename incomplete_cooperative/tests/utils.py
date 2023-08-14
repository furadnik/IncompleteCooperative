"""Utils for testing."""
from incomplete_cooperative.bounds import compute_bounds_superadditive
from incomplete_cooperative.coalitions import Coalition, all_coalitions
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.icg_gym import ICG_Gym
from incomplete_cooperative.protocols import MutableIncompleteGame


def trivial_fill(game: MutableIncompleteGame) -> None:
    """Trivially fill game."""
    for coalition in all_coalitions(game):
        game.set_value(len(coalition)**2, coalition)


class IncompleteGameMixin:
    """Mixin for testing incomplete game."""

    def get_game(self, *, number_of_players=6) -> IncompleteCooperativeGame:
        """Get game based on params."""
        return IncompleteCooperativeGame(number_of_players, compute_bounds_superadditive)


class GymMixin(IncompleteGameMixin):
    """Mixin for testing Gym."""

    def get_gym(self, *, known_coalitions=[1, 2, 4, 8, 16, 32, 63], **kwargs) -> ICG_Gym:
        """Get gym based on params."""
        incomplete_game = self.get_game(**kwargs)
        full_game = self.get_game(**kwargs)
        trivial_fill(full_game)
        transformed_known_coalitions = list(map(Coalition, filter(lambda x: x < 2**incomplete_game.number_of_players,
                                                                  known_coalitions)))
        return ICG_Gym(incomplete_game, lambda: full_game, transformed_known_coalitions)
