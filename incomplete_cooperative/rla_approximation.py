"""Root Linear Approximation of a cooperative game."""
from math import sqrt

import numpy as np

from .coalitions import Coalition, all_coalitions, player_to_coalition
from .game import IncompleteCooperativeGame
from .protocols import IncompleteGame


def compute_rla_approximation(game: IncompleteGame,
                              epsilon: float = 0.05) -> tuple[np.ndarray, IncompleteCooperativeGame]:
    """Compute a sketch of a cooperative game using a root linear function, i.e. sqrt(c_1 x_1 + ... c_n x_n).

    - c_i ... weight of agent i
    - x_i ... indicator variable of player i
    - f(S) = sum_{i_in_S} c_i x_i.
    """
    weights, queried_coalition_ids = _compute_weights_and_queries(game, epsilon)

    approximated_values = _get_approximated_values(game.number_of_players, weights)
    approximated_game = IncompleteCooperativeGame(game.number_of_players)
    approximated_game.set_values(approximated_values)

    return queried_coalition_ids, approximated_game


def _compute_weights_and_queries(game: IncompleteGame, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the sketch of the game using the ASFE algorithm.

    The game is assumed to be subadditive and monotone.
    """
    queried_coalition_ids = np.array([])
    weights = np.array([game.number_of_players / game.get_value(player_to_coalition(i)) ** 2
                                 for i in range(game.number_of_players)])

    while True:
        
        vector, new_queried_coalition_ids = _approximate_max_on_polymatroid(game, weights)
        queried_coalition_ids = np.concatenate([queried_coalition_ids, new_queried_coalition_ids])

        weight_matrix = np.diag(weights)
      
        if sqrt((vector.T @ weight_matrix @ vector).item()) > (
            sqrt(game.number_of_players) + epsilon
        ):
            temp_matrix = _compute_larger_ellipsoid(game.number_of_players, weight_matrix, vector)

            inverted_matrix = np.linalg.inv(temp_matrix)
            weights = 1 / np.diag(inverted_matrix)

        else:
            break

    return weights, np.unique(queried_coalition_ids)


def _approximate_max_on_polymatroid(game: IncompleteGame, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximates the maximum | |x||_weight_matrix on the polymatroid given by the set function.

    More specifically, the maximum is approximated for a modified set function

        g(S) = c_1 * [f([2, k]) - f([1, k])] + ... + c_k[f([k, k]) - f([k - 1, k])].

    The maximum is then attained greedily by adding the player, which maximizes the marginal gain.
    """
    queried_coalition_ids = np.array([])
    res_vector = np.zeros(game.number_of_players)

    coalition = Coalition(0)
    complement_coalition = coalition.inverted(game.number_of_players)

    last_max = 0
    for _ in range(game.number_of_players):
        temp_max = 0
        for player in complement_coalition.players:
            g_value, new_queried_coalition_ids = _query_values_and_compute_g_function(
                game, coalition + player, weights
            )
            queried_coalition_ids = np.concatenate([queried_coalition_ids, new_queried_coalition_ids])
            if g_value > temp_max:
                temp_max = g_value
                player_to_add = player

        res_vector[player_to_add] = temp_max - last_max
        coalition += player_to_add
        complement_coalition -= player_to_add
        last_max = temp_max

    return res_vector.reshape(-1, 1), queried_coalition_ids


def _compute_larger_ellipsoid(num_of_players: int, weight_matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Based on a matrix representation of an ellipsoid and vector, computes a larger ellipsoid(its matrix).

    Formula from Lemma 2 in paper.
    """
    norm_sqr = vector.T @ weight_matrix @ vector
    frac = (norm_sqr - 1) / (num_of_players - 1)

    larger_ellipsoid = (num_of_players / norm_sqr * frac * weight_matrix) + (
        (num_of_players / (norm_sqr**2)) * (1 - (norm_sqr - 1) / (num_of_players - 1)) * (
            weight_matrix @ vector @ vector.T @ weight_matrix
        )
    )
    return larger_ellipsoid


def _query_values_and_compute_g_function(game: IncompleteGame, coalition: Coalition,
                                         weights: np.ndarray) -> tuple[int, np.ndarray]:
    """Return the value of the function g(S) = c_1 * [f([2, k]) - f([1, k])] + ... + c_k[f([k, k]) - f([k - 1, k])]."""
    queried_coalition_ids = np.array([])
    coalition_weights = weights[list(coalition.players)]
    sorted_indices = np.argsort(coalition_weights)[::-1]

    result = 0
    temp_id = 0

    for i in sorted_indices:
        old_temp_id = temp_id
        temp_id += 2**i
        queried_coalition_ids = np.append(queried_coalition_ids, temp_id)
        result += weights[i] * (game.get_value(Coalition(temp_id)) - game.get_value(Coalition(old_temp_id)))
    return result, np.array(queried_coalition_ids)


def _get_value(coalition: Coalition, weights: np.ndarray) -> float:
    """Return value of the Root Linear approximation for a given coalition."""
    return sqrt(sum(weights[i] for i in coalition.players))


def _get_approximated_values(num_of_players: int, weights: np.ndarray) -> np.ndarray:
    """Return the approximation of the game given the weights."""
    return np.array([_get_value(coalition, weights) for coalition in all_coalitions(num_of_players)])
