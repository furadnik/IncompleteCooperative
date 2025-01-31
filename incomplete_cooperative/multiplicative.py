"""Existing algorithms for computing sketches with multiplicative bounds guarantees.

    There are two algorithm from two papers:
    1) Root Linear Approximation - approximation from paper Approximate Submodular Function Everywhere: for submodular monotone functions only
    2) Max Xos Approximation - approximation from paper Simpler and faster algorithm: for both submodular monotone and subadditive functions

"""

import numpy as np
from math import log, sqrt, log2, floor

from coalitions import (
    Coalition,
    all_coalitions,
    player_to_coalition,
)
from game import IncompleteCooperativeGame


class RootLinearApproximation:
    """Represents a sketch of a cooperative game using a root linear function, i.e. sqrt(c_1 x_1 + ... c_n x_n) where
    - c_i ... weight of agent i
    - x_i ... indicator variable of player i
    - f(S) = sum_{i_in_S} c_i x_i.
    """

    def __init__(self, game: IncompleteCooperativeGame, epsilon: float = 0.05):
        """Initializes the Root Linear Approximation of a cooperative game, computing the weights and storing value queries."""
        self.number_of_players = game.number_of_players
        self.values = game._values[:, 1]

        self.queried_coalitions = []  # Computed in self.compute_weights
        self.weights = self.compute_weights(game._values, epsilon)

        # Remove duplicates
        self.queried_coalitions = np.array(list(set(self.queried_coalitions)))

        self.multiplicative_factor = self.compute_multiplicative_factor()

    def get_value(self, coalition: int) -> int:
        """Return value of the Root Linear approximation for a given coalition."""
        return sqrt(sum(self.weights[i] for i in coalition.players))

    def get_g(self, coalition: Coalition, weights_matrix: np.array) -> int:
        """Returns the value of the function g(S) = c_1 * [f([2,k]) - f([1,k])] + ... + c_k[f([k,k]) - f([k-1,k])]."""

        bin_repr_coalition_id = bin(coalition.id)[2:].zfill(self.number_of_players)

        filter_array = [bit == "1" for bit in reversed(bin_repr_coalition_id)]

        weights = np.diag(weights_matrix)
        coalition_weights = weights[filter_array]

        sorted_indices = np.argsort(coalition_weights)[::-1]

        result = 0
        temp_id = 0

        for i in sorted_indices:
            old_temp_id = temp_id

            temp_id += 2**i

            self.queried_coalitions.append(
                temp_id
            )  # Here, values are queried
            result += weights[i] * (self.values[temp_id] - self.values[old_temp_id])
        return result

    def approximate_max_on_polymatroid(self, weights_matrix) -> np.array:
        """Approximates the maximum ||x||_weight_matrix on the polymatroid given by the set function.
        More specifically, the maximum is approximated for a modified set function g(S) = c_1 * [f([2,k]) - f([1,k])] + ... + c_k[f([k,k]) - f([k-1,k])].
        The maximum is then attained greedily by adding the player, which maximizes the marginal gain and setting the result vector using these .
        """
        res_vector = np.zeros(self.number_of_players)

        coalition = Coalition(0)
        complement_coalition = coalition.inverted(self.number_of_players)

        last_max = 0
        for i in range(self.number_of_players):
            temp_max = 0
            for player in complement_coalition.players:
                g_value = self.get_g(coalition + player, weights_matrix)
                if g_value > temp_max:
                    temp_max = g_value
                    player_to_add = player

            res_vector[player_to_add] = temp_max - last_max
            coalition += player_to_add
            complement_coalition -= player_to_add
            last_max = temp_max

        return res_vector.reshape(-1, 1)

    def compute_larger_ellipsoid(self, matrix: np.array, vector: np.array) -> np.array:
        """Based on a matrix representation of an ellipsoid and vector, computes a larger ellipsoid (its matrix). Formula from Lemma 2 in paper."""

        n = self.number_of_players
        norm_sqr = vector.T @ matrix @ vector
        frac = (norm_sqr - 1) / (n - 1)

        larger_ellipsoid = (n / norm_sqr * frac * matrix) + (
            (n / (norm_sqr**2))
            * (1 - (norm_sqr - 1) / (n - 1))
            * (matrix @ vector @ vector.T @ matrix)
        )
        return larger_ellipsoid

    def compute_weights(self, values: np.array, epsilon: float):
        """Computes the sketch of the game using the ASFE algorithm."""

        diagonal_entries = np.array(
            [
                (self.number_of_players / self.values[2**i]) ** 2
                for i in range(self.number_of_players)
            ]
        )

        # Weights are iteratively computed and stored on diagonal
        weight_matrix = np.diag(diagonal_entries)

        while True:
            vector = self.approximate_max_on_polymatroid(weight_matrix)

            if sqrt((vector.T @ weight_matrix @ vector).item()) > (
                sqrt(self.number_of_players) + epsilon
            ):
                temp_matrix = self.compute_larger_ellipsoid(weight_matrix, vector)

                weight_matrix = np.linalg.inv(
                    np.diag(np.diag(np.linalg.inv(temp_matrix)))
                )  # diag is twice to get diag matrix, not just vector of diag elements
            else:
                break

        return np.diag(weight_matrix)

    def compute_multiplicative_factor(self):
        """Computes the multiplicative factor of the approximation. It is aprox(S) / v(S) <= alpha, as opposed to the other algorithm."""
        multiplicative_factor = 0
        for coalition in all_coalitions(self.number_of_players):
            if coalition.id == 0:
                continue
            multiplicative_factor = max(
                multiplicative_factor,
                self.get_value(coalition) / self.values[coalition.id],
            )
        return multiplicative_factor


class MaxXosApproximation:
    def __init__(
        self, game: IncompleteCooperativeGame, alpha: float = 3.7844223824, beta: float = 1, eps: float = 0.05
    ):
        self.number_of_players = game.number_of_players
        self.values = game._values[:, 1]
        # self.values = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5])  # For testing purposes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.queried_coalitions = []
        self.approximation = self.construct_approximation()

        # Remove duplicates
        self.queried_coalitions = np.array(list(set(self.queried_coalitions)))
        self.multiplicative_factor = self.compute_multiplicative_factor()

    def construct_approximation(self):
        """Constructs the approximation of the game using the Max XOS algorithm."""

        k_values, r_values = self.get_k_r_values()

        candidate_coalitions = np.empty((len(k_values), len(r_values)), dtype=object)

        heavy_players = np.zeros((len(k_values), len(r_values), self.number_of_players))
        light_players = np.zeros((len(k_values), len(r_values), self.number_of_players))

        singleton_ids = 2 ** np.arange(self.number_of_players)
        singleton_values = self.values[singleton_ids]

        # Constructing candidate_coalitions, translate to numpy for more efficiency
        for k in range(len(k_values)):
            for r in range(len(r_values)):

                candidate_coalitions[k, r] = []

                heavy_players[k, r, :] = singleton_values >= (
                    k_values[k] * r_values[r] / sqrt(self.number_of_players)
                )
                light_players[k, r] = 1 - heavy_players[k, r]

                remaining_players = light_players[k, r].astype(bool)

                # Translating remaining_players to a Coalition
                remaining_coalition = Coalition(singleton_ids[remaining_players].sum())
                coalition = self.max_subroutine(remaining_coalition, k_values[k])
                while self.values[coalition.id] >= k_values[k] * r_values[r] / (2 * self.alpha):

                    additive_vector = self.approx_xos_subroutine(coalition)
                    subcoalition = coalition

                    for player in coalition.players:
                        if not additive_vector[player] >= r_values[r] / (4 * self.alpha * self.beta):
                            subcoalition -= player_to_coalition(player)

                    candidate_coalitions[k, r].append(subcoalition)

                    coalition = self.max_subroutine(
                        coalition - subcoalition, k_values[k]
                    )

        # Constructing the approximation based on candidate_coalitions
        approximation = np.zeros(2**self.number_of_players)

        for coalition in all_coalitions(self.number_of_players):
            if coalition.id != 0:
                max_singleton = np.max(singleton_values[list(coalition.players)])

                max_value = max_singleton
                for k in range(len(k_values)):
                    for r in range(len(r_values)):
                        for candidate_coalition in candidate_coalitions[k, r]:
                            new_value = (
                                len(candidate_coalition & coalition)
                                * r
                                / (4 * self.alpha * self.beta)
                            )
                            if new_value > max_value:
                                max_value = new_value
                approximation[coalition.id] = max_value
        return approximation

    def max_subroutine(self, coalition: Coalition, size: int) -> Coalition:
        """Computes approximate solution of k-bounded max problem for set function restricted to coalition."""

        if coalition == Coalition(0):
            return Coalition(0)
        else:
            singleton_ids = 2 ** np.arange(self.number_of_players)
            singleton_values = self.values[singleton_ids]

            initial_limit = np.max(singleton_values[list(coalition.players)])
            constructed_coalition = Coalition(0)  # Empty coalition
            reduced_limit = initial_limit
            while reduced_limit >= self.eps * initial_limit / self.number_of_players:
                for player in (coalition - constructed_coalition).players:
                    if len(constructed_coalition) + 1 < size:
                        self.queried_coalitions.append(
                            (constructed_coalition + player).id
                        )  # Here, values are queried
                        if (
                            self.values[(constructed_coalition + player).id]
                            - self.values[constructed_coalition.id]
                            >= reduced_limit
                        ):
                            constructed_coalition += player
                    else:
                        return constructed_coalition

                reduced_limit = reduced_limit * (1 - self.eps)

            return constructed_coalition

    def approx_xos_subroutine(self, coalition: Coalition) -> np.ndarray:
        """Computes beta-XOS clause for coalition with respect to valuation of the game."""

        additive_vector = np.zeros(self.number_of_players)

        subcoalition = Coalition(0)
        extended_subcoalition = Coalition(0)

        for player in coalition.players:
            subcoalition = extended_subcoalition
            extended_subcoalition += player
            self.queried_coalitions.append(extended_subcoalition.id)  # Here, values are queried

            additive_vector[player] = (self.values[extended_subcoalition.id] - self.values[subcoalition.id])

        return additive_vector

    def get_k_r_values(self):
        """Computes parameters of the algorith. I am too exhausted to write it in a better way."""
        k_values = []
        k = 0
        square = sqrt(self.number_of_players)
        while (2**k) * square < self.number_of_players:
            k_values.append(2**k * square)
            k += 1
        k_values.append(self.number_of_players)

        r_values = []
        r = 0
        while 2**r < self.number_of_players**2:
            r_values.append(2**r)
            r += 1
        r_values.append(self.number_of_players**2)
        return k_values, r_values

    def compute_multiplicative_factor(self):
        """Computes the multiplicative factor of the approximation. It is v(S) / aprox(S) <= alpha, as opposed to the other algorithm."""
        multiplicative_factor = 0
        for coalition in all_coalitions(self.number_of_players):
            multiplicative_factor = max(
                multiplicative_factor,
                self.values[coalition.id] / self.approximation[coalition.id],
            )
        return multiplicative_factor
