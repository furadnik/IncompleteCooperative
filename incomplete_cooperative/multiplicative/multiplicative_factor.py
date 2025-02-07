"""Compute multiplicative factors."""
import numpy as np

from ..protocols import Game, IncompleteGame


def mul_factor_to_approximation(game: Game, approximated_game: Game) -> float:
    """Compute the multiplicative factor of the approximation.

    It is the smallest alpha, such that v(S) / aprox(S) <= alpha, as opposed to the other algorithm.
    """
    approximated_values = approximated_game.get_values()
    original_values = game.get_values()

    # we want to compare only where the original values aren't zero, as we'd get an error...
    original_values_nonzero = original_values != 0

    return np.max(approximated_values[original_values_nonzero] / original_values[original_values_nonzero])


def mul_factor_to_lower_bound(game: Game, incomplete_game: IncompleteGame) -> float:
    """Compute the multiplicative factor of the lower bound.

    It is the smallest alpha, such that v(S) / lb(S) <= alpha.
    """
    incomplete_game.compute_bounds()

    lower_bounds = incomplete_game.get_lower_bounds()
    original_values = game.get_values()

    # we want to compare only where the original values aren't zero, as we'd get an error...
    original_values_nonzero = original_values != 0

    return np.max(lower_bounds[original_values_nonzero] / original_values[original_values_nonzero])


def mul_factor_lower_upper_bound(incomplete_game: IncompleteGame) -> float:
    """Compute the multiplicative factor of the lower bound.

    It is the smallest alpha, such that v(S) / lb(S) <= alpha.
    """
    incomplete_game.compute_bounds()

    lower_bounds = incomplete_game.get_lower_bounds()
    upper_bounds = incomplete_game.get_upper_bounds()

    # we don't want to divide by zero
    upper_bounds_nonzero = upper_bounds != 0

    return np.max(lower_bounds[upper_bounds_nonzero] / upper_bounds[upper_bounds_nonzero])
