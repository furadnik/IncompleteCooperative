"""Compute multiplicative factors."""
import numpy as np

from ..protocols import Game, IncompleteGame


def mul_factor_to_approximation(game: Game, approximated_game: Game) -> float:
    """Compute the multiplicative factor of the approximation.

    It is the smallest alpha, such that v(S) / aprox(S) <= alpha, as opposed to the other algorithm.
    """
    approximated_values = approximated_game.get_values()[1:]
    original_values = game.get_values()[1:]

    assert np.all(original_values >= approximated_values)
    assert np.all(approximated_values > 0)
    return np.max(original_values / approximated_values)


def mul_factor_upper_to_approximation(approximated_game: Game, incomplete_game: IncompleteGame) -> float:
    """Compute the multiplicative factor of the approximation.

    It is the smallest alpha, such that ub(S) / aprox(S) <= alpha, as opposed to the other algorithm.
    """
    approximated_values = approximated_game.get_values()[1:]
    upper_values = incomplete_game.get_upper_bounds()[1:]

    assert np.all(upper_values >= approximated_values)
    assert np.all(approximated_values > 0)
    return np.max(upper_values / approximated_values)


def mul_factor_to_lower_bound(game: Game, incomplete_game: IncompleteGame) -> float:
    """Compute the multiplicative factor of the lower bound.

    It is the smallest alpha, such that v(S) / lb(S) <= alpha.
    """
    lower_bounds = incomplete_game.get_lower_bounds()[1:]
    original_values = game.get_values()[1:]

    assert np.all(original_values >= lower_bounds)
    assert np.all(lower_bounds > 0)
    return np.max(original_values / lower_bounds)


def mul_factor_lower_upper_bound(incomplete_game: IncompleteGame) -> float:
    """Compute the multiplicative factor of the lower bound.

    It is the smallest alpha, such that ub(S) / lb(S) <= alpha.
    """
    lower_bounds = incomplete_game.get_lower_bounds()[1:]
    upper_bounds = incomplete_game.get_upper_bounds()[1:]

    assert np.all(upper_bounds >= lower_bounds), f"upper_bounds={upper_bounds}, lower_bounds={lower_bounds}"
    assert np.all(lower_bounds > 0), f"known={incomplete_game.get_known_values()},upper_bounds={upper_bounds}, lower_bounds={lower_bounds}"
    return np.max(upper_bounds / lower_bounds)
