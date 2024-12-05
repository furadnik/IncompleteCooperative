"""In this module, Coalition computations are performed in numpy using coalition ids."""
from typing import Any

import numpy as np

CoalitionId = np.int32


def get_all_coalitions(number_of_players: int | np.integer) -> np.ndarray[Any, np.dtype[CoalitionId]]:
    """Get the set of all coalitions for a game."""
    return np.arange(2**number_of_players, dtype=CoalitionId)


def players(coalition: CoalitionId, number_of_players: int) -> np.ndarray[Any, np.dtype[np.int32]]:
    """Get players in a coalition.

    number_of_players is the total number of players in the game.
    """
    assert 2**number_of_players > coalition, "Coalition has more players than number_of_players allows."
    return np.arange(number_of_players)[2**np.arange(number_of_players) & coalition != 0]


def get_size(coalition: CoalitionId, number_of_players: int) -> np.ndarray[Any, np.dtype[np.int32]]:
    """Get players in a coalition.

    number_of_players is the total number of players in the game.
    """
    assert 2**number_of_players > coalition, f"Coalition {coalition} has more players than number_of_players allows - {number_of_players}."
    return (2**np.arange(number_of_players) & coalition != 0).sum()


def sub_coalitions(coalition: CoalitionId, number_of_players: int) -> np.ndarray[Any, np.dtype[CoalitionId]]:
    """Get subcoalitions of a coalition."""
    assert 2**number_of_players > coalition, f"Coalition {coalition} has more players than number_of_players allows - {number_of_players}."
    max_num_players = np.max(players(coalition, number_of_players), initial=0) + 1
    return get_all_coalitions(max_num_players)[get_all_coalitions(max_num_players) | coalition == coalition]


def super_coalitions(coalition: CoalitionId, number_of_players: int) -> np.ndarray[Any, np.dtype[CoalitionId]]:
    """Get subcoalitions of a coalition."""
    assert 2**number_of_players > coalition, f"Coalition {coalition} has more players than number_of_players allows - {number_of_players}."

    opposite_coalition = (2**number_of_players - 1) ^ coalition
    opposite_sub = sub_coalitions(opposite_coalition, number_of_players)
    return opposite_sub | coalition
