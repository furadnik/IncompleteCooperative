"""Norms on games."""
from functools import partial

import numpy as np

from incomplete_cooperative.protocols import IncompleteGame, Value


def lp_norm(game: IncompleteGame, ord: int | float | None = None) -> Value:
    """Compute an lp norm on vectors."""
    return Value(np.linalg.norm((game.get_upper_bounds() - game.get_lower_bounds()), ord))


l2_norm = partial(lp_norm, ord=2)
l1_norm = partial(lp_norm, ord=1)
linf_norm = partial(lp_norm, ord=np.inf)
