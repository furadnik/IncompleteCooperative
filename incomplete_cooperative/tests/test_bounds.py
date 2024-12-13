import random
from functools import partial

import numpy as np
import pytest

from incomplete_cooperative.bounds import (
    compute_bounds_superadditive, compute_bounds_superadditive_cached,
    compute_bounds_superadditive_monotone_approx_cached)
from incomplete_cooperative.coalitions import (Coalition, all_coalitions,
                                               grand_coalition,
                                               minimal_game_coalitions)
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.generators import (convex_generator,
                                               covg_fn_generator,
                                               k_budget_generator)


def dummy_fill(game: IncompleteCooperativeGame) -> None:
    """Compute dummy values."""
    for coalition in all_coalitions(game):
        game.set_value(len(coalition), coalition)


NUMBER_OF_PLAYERS = 6
SAM_GENERATORS = [covg_fn_generator, k_budget_generator]


@pytest.fixture
def game():
    return IncompleteCooperativeGame(NUMBER_OF_PLAYERS, compute_bounds_superadditive)


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


def get_game_sam_apx(rep):
    return IncompleteCooperativeGame(
        NUMBER_OF_PLAYERS,
        partial(compute_bounds_superadditive_monotone_approx_cached, repetitions=rep))


@pytest.mark.parametrize("game_sam_apx", [get_game_sam_apx(rep) for rep in range(10)])
def test_full_game_sam_apx(game_sam_apx):
    dummy_fill(game_sam_apx)
    game_sam_apx.compute_bounds()
    for coalition in all_coalitions(game_sam_apx):
        assert game_sam_apx.get_value(coalition) == game_sam_apx.get_upper_bound(coalition)
        assert game_sam_apx.get_value(coalition) == game_sam_apx.get_lower_bound(coalition)


@pytest.mark.parametrize("rep", range(100))
@pytest.mark.parametrize("gen", SAM_GENERATORS)
def test_sam_apx_stronger_than_sa(game, rep, gen):
    """Test that the SAM bounds always give a stronger bound than simple SA bounds."""
    games_sam_apx = [get_game_sam_apx(rep) for rep in range(10)]
    assert all(game_sam_apx.number_of_players == game.number_of_players for game_sam_apx in games_sam_apx)
    random_sam = gen(game.number_of_players)
    for game_sam_apx in games_sam_apx:
        game_sam_apx.set_value(random_sam.get_value(grand_coalition(random_sam)), grand_coalition(game_sam_apx))
    game.set_value(random_sam.get_value(grand_coalition(random_sam)), grand_coalition(game_sam_apx))
    for player in range(game_sam_apx.number_of_players):
        for game_sam_apx in games_sam_apx:
            game_sam_apx.set_value(random_sam.get_value(Coalition.from_players([player])), Coalition.from_players([player]))
        game.set_value(random_sam.get_value(Coalition.from_players([player])), Coalition.from_players([player]))
    for coalition in all_coalitions(game):
        if (not game.is_value_known(coalition)) and random.randint(0, 1) == 1:  # nosec
            for game_sam_apx in games_sam_apx:
                game_sam_apx.set_value(random_sam.get_value(coalition), coalition)
            game.set_value(random_sam.get_value(coalition), coalition)
    for game_sam_apx in games_sam_apx:
        game_sam_apx.compute_bounds()
    game.compute_bounds()

    for coalition in all_coalitions(game_sam_apx):
        for i, game_sam_apx in enumerate(games_sam_apx):
            assert game_sam_apx.get_lower_bound(coalition) >= game.get_lower_bound(coalition)
            assert game_sam_apx.get_upper_bound(coalition) <= game.get_upper_bound(coalition)
            if i > 0:
                prev_sam_apx = games_sam_apx[i - 1]
                assert game_sam_apx.get_upper_bound(coalition) <= prev_sam_apx.get_upper_bound(coalition)
                assert game_sam_apx.get_lower_bound(coalition) >= prev_sam_apx.get_lower_bound(coalition)
