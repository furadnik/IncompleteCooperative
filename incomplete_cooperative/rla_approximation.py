import numpy as np
from math import sqrt

from .coalitions import Coalition, all_coalitions, player_to_coalition
from .game import IncompleteCooperativeGame


def compute_rla_approximation(game: IncompleteCooperativeGame, epsilon: float = 0.05) -> tuple[np.array, float]:
    """Computes a sketch of a cooperative game using a root linear function, i.e. sqrt(c_1 x_1 + ... c_n x_n) where
    - c_i ... weight of agent i
    - x_i ... indicator variable of player i
    - f(S) = sum_{i_in_S} c_i x_i.
    """
    
    weights, queried_coalition_ids = compute_weights_and_queries(game, epsilon)
    multiplicative_factor = compute_multiplicative_factor(game, weights)

    return queried_coalition_ids, multiplicative_factor


def compute_weights_and_queries(game: IncompleteCooperativeGame, epsilon: float) -> tuple[np.array, np.array]:
    """Computes the sketch of the game using the ASFE algorithm."""

    queried_coalition_ids = []

    diagonal_entries = np.array([game.number_of_players / game.get_value(player_to_coalition(i)) ** 2 for i in range(game.number_of_players)])

    # Weights are iteratively computed and stored on diagonal
    weight_matrix = np.diag(diagonal_entries)

    while True:
        vector, new_queried_coalition_ids = approximate_max_on_polymatroid(game.number_of_players, weight_matrix)
        queried_coalition_ids = np.concatenate(queried_coalition_ids, new_queried_coalition_ids)

        if sqrt((vector.T @ weight_matrix @ vector).item()) > (
            sqrt(game.number_of_players) + epsilon
        ):
            temp_matrix = compute_larger_ellipsoid(game.number_of_players, weight_matrix, vector)

            weight_matrix = np.linalg.inv(np.diag(np.diag(np.linalg.inv(temp_matrix))))  # diag is twice to get diag matrix, not just vector of diag elements
        else:
            break

    return np.diag(weight_matrix), np.unique(queried_coalition_ids)

def approximate_max_on_polymatroid(game : IncompleteCooperativeGame, weights_matrix: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Approximates the maximum ||x||_weight_matrix on the polymatroid given by the set function.
    More specifically, the maximum is approximated for a modified set function g(S) = c_1 * [f([2,k]) - f([1,k])] + ... + c_k[f([k,k]) - f([k-1,k])].
    The maximum is then attained greedily by adding the player, which maximizes the marginal gain.
    """
    queried_coalition_ids = []
    res_vector = np.zeros(game.number_of_players)

    coalition = Coalition(0)
    complement_coalition = coalition.inverted(game.number_of_players)

    last_max = 0
    for _ in range(game.number_of_players):
        temp_max = 0
        for player in complement_coalition.players:
            g_value, new_queried_coalition_ids = query_values_and_compute_g_function(
                game, coalition + player, weights_matrix
                )
            queried_coalition_ids = np.concatenate(queried_coalition_ids, new_queried_coalition_ids)
            if g_value > temp_max:
                temp_max = g_value
                player_to_add = player

        res_vector[player_to_add] = temp_max - last_max
        coalition += player_to_add
        complement_coalition -= player_to_add
        last_max = temp_max

    return res_vector.reshape(-1, 1), queried_coalition_ids

def compute_larger_ellipsoid(num_of_players : int, weight_matrix: np.array, vector: np.array) -> np.array:
    """Based on a matrix representation of an ellipsoid and vector, computes a larger ellipsoid (its matrix). Formula from Lemma 2 in paper."""

    
    norm_sqr = vector.T @ weight_matrix @ vector
    frac = (norm_sqr - 1) / (num_of_players - 1)

    larger_ellipsoid = (num_of_players / norm_sqr * frac * weight_matrix) + (
        (num_of_players / (norm_sqr**2))
        * (1 - (norm_sqr - 1) / (num_of_players - 1))
        * (weight_matrix @ vector @ vector.T @ weight_matrix)
    )
    return larger_ellipsoid

def query_values_and_compute_g_function(game : IncompleteCooperativeGame, coalition: Coalition, weights_matrix: np.array) -> tuple[int, np.ndarray]:
    """Returns the value of the function g(S) = c_1 * [f([2,k]) - f([1,k])] + ... + c_k[f([k,k]) - f([k-1,k])]."""

    queried_coalition_ids = []

    weights = np.diag(weights_matrix)

    coalition_weights = weights[list(coalition.players)]
    sorted_indices = np.argsort(coalition_weights)[::-1]

    result = 0
    temp_id = 0

    for i in sorted_indices:
        old_temp_id = temp_id
        temp_id += 2**i
        queried_coalition_ids.append(temp_id)
        result += weights[i] * (game.get_value(Coalition(temp_id)) - game.get_value(Coalition(old_temp_id)))
    return result, queried_coalition_ids

def get_value(coalition: Coalition, weights: np.ndarray) -> float:
    """Return value of the Root Linear approximation for a given coalition."""
    return sqrt(sum(weights[i] for i in coalition.players))

def compute_multiplicative_factor(game : IncompleteCooperativeGame, weights: np.array) -> float:
    """Computes the multiplicative factor of the approximation. It is aprox(S) / v(S) <= alpha, as opposed to the other algorithm."""
    multiplicative_factor = 0
    for coalition in all_coalitions(game.number_of_players):
        if coalition.id == 0:
            continue
        multiplicative_factor = max(
            multiplicative_factor,
            get_value(coalition, weights) / game.get_value(Coalition(coalition.id)),
        )
    return multiplicative_factor
