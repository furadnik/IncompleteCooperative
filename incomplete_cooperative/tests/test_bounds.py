import numpy as np
import pytest

from incomplete_cooperative.bounds import (compute_bounds_superadditive,
                                           compute_bounds_superadditive_cached)
from incomplete_cooperative.coalitions import (Coalition, all_coalitions,
                                               grand_coalition,
                                               minimal_game_coalitions)
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.generators import convex_generator


def dummy_fill(game: IncompleteCooperativeGame) -> None:
    """Compute dummy values."""
    for coalition in all_coalitions(game):
        game.set_value(len(coalition), coalition)


NUMBER_OF_PLAYERS = 6


@pytest.fixture
def game():
    return IncompleteCooperativeGame(6, compute_bounds_superadditive)


def test_full_game(game):
    dummy_fill(game)
    game.compute_bounds()
    for coalition in all_coalitions(game):
        assert game.get_value(coalition) == game.get_upper_bound(coalition)
        assert game.get_value(coalition) == game.get_lower_bound(coalition)


def test_minimal_game(game):
    game.set_value(2 * game.number_of_players, grand_coalition(game))
    for player in range(game.number_of_players):
        game.set_value(1, Coalition.from_players([player]))
    game.compute_bounds()

    for coalition in filter(game.is_value_known, all_coalitions(game)):
        assert game.get_upper_bound(coalition) == game.get_value(coalition)
        assert game.get_lower_bound(coalition) == game.get_value(coalition)

    for coalition in filter(lambda x: not game.is_value_known(x), all_coalitions(game)):
        assert game.get_upper_bound(coalition) == game.number_of_players + len(coalition)
        assert game.get_lower_bound(coalition) == len(coalition)


@pytest.mark.parametrize("rep", range(10))
def test_cached_nocached_same(rep):
    min_game = list(minimal_game_coalitions(NUMBER_OF_PLAYERS))
    incomplete_game = IncompleteCooperativeGame(NUMBER_OF_PLAYERS, compute_bounds_superadditive)
    incomplete_game_cached = IncompleteCooperativeGame(NUMBER_OF_PLAYERS, compute_bounds_superadditive_cached)
    game = convex_generator(NUMBER_OF_PLAYERS)
    incomplete_game.set_known_values(game.get_values(min_game), min_game)
    incomplete_game_cached.set_known_values(game.get_values(min_game), min_game)
    incomplete_game.compute_bounds()
    incomplete_game_cached.compute_bounds()
    assert np.allclose(incomplete_game_cached.get_upper_bounds(), incomplete_game.get_upper_bounds())
    assert np.allclose(incomplete_game_cached.get_lower_bounds(), incomplete_game.get_lower_bounds())
    for step in range(2**game.number_of_players - len(min_game)):
        unknown = np.arange(2**NUMBER_OF_PLAYERS)[np.logical_not(incomplete_game.are_values_known())]
        random = np.random.choice(unknown)
        incomplete_game.reveal_value(game.get_value(Coalition(random)), Coalition(random))
        incomplete_game.compute_bounds()
        incomplete_game_cached.reveal_value(game.get_value(Coalition(random)), Coalition(random))
        incomplete_game_cached.compute_bounds()
        assert np.allclose(incomplete_game_cached.get_upper_bounds(), incomplete_game.get_upper_bounds())
        assert np.allclose(incomplete_game_cached.get_lower_bounds(), incomplete_game.get_lower_bounds())
