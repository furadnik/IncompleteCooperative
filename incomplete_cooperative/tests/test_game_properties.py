"""Test whether the game properties are accurate."""
import numpy as np

from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.game_properties import (is_monotone_decreasing,
                                                    is_sam, is_superadditive)


def test_superadditivity_true():
    values = [0, 1, 1, 3, 0, 2, 2, 4]
    game = IncompleteCooperativeGame(3)
    game.set_values(np.array(values))
    assert is_superadditive(game)


def test_superadditivity_false():
    values = [0, 3, 1, 3, 0, 2, 2, 4]
    game = IncompleteCooperativeGame(3)
    game.set_values(np.array(values))
    assert not is_superadditive(game)


def test_monotone_true():
    values = [0, -1, -2, -3, -4, -5, -6, -7]
    game = IncompleteCooperativeGame(3)
    game.set_values(np.array(values))
    assert is_monotone_decreasing(game)


def test_monotone_false():
    values = [0, 3, 1, 3, 0, 2, 2, 4]
    game = IncompleteCooperativeGame(3)
    game.set_values(np.array(values))
    assert not is_monotone_decreasing(game)


def test_sam_true():
    values = [0, -1, -1, -2, -1, -2, -2, -2]
    game = IncompleteCooperativeGame(3)
    game.set_values(np.array(values))
    assert is_superadditive(game)
    assert is_monotone_decreasing(game)
    assert is_sam(game)


def test_sam_false_sa():
    values = [0, 3, 1, 3, 0, 2, 2, 4]
    game = IncompleteCooperativeGame(3)
    game.set_values(np.array(values))
    assert not is_sam(game)


def test_sam_false_mon():
    values = [0, 3, 1, 3, 0, 2, 2, 4]
    game = IncompleteCooperativeGame(3)
    game.set_values(np.array(values))
    assert not is_sam(game)
