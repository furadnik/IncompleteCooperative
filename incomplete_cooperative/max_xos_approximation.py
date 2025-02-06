import numpy as np
from math import sqrt

from .coalitions import (
    Coalition,
    all_coalitions,
    player_to_coalition,
)
from .game import IncompleteCooperativeGame


def compute_max_xos_approximation(game : IncompleteCooperativeGame, alpha: float = 3.7844223824, beta: float = 1, eps: float = 0.05) -> tuple[np.array, float]:
    k_values, r_values = get_k_r_values(game.number_of_players)

    candidate_coalitions, queried_coalition_ids = compute_candidate_coalitions_and_query_values(
        game, k_values, r_values, alpha, beta, eps
        )
    
    approximated_values = compute_approximation(game, candidate_coalitions, k_values, r_values, alpha, beta)
    approximated_game = IncompleteCooperativeGame(game.number_of_players)
    approximated_game.set_values(approximated_values)
    
    #multiplicative_factor = compute_multiplicative_factor(game, approximated_values)

    return queried_coalition_ids, approximated_game#, multiplicative_factor



def get_k_r_values(number_of_players: int) -> tuple[list[int], list[int]]: #TODO: Rewrite, ugly
    """Computes parameters of the algorithm."""
    k_values = []
    k = 0
    square = sqrt(number_of_players)
    while (2**k) * square < number_of_players:
        k_values.append(2**k * square)
        k += 1
    k_values.append(number_of_players)

    r_values = []
    r = 0
    while 2**r < number_of_players**2:
        r_values.append(2**r)
        r += 1
    r_values.append(number_of_players**2)
    return k_values, r_values

def compute_candidate_coalitions_and_query_values(game: IncompleteCooperativeGame, k_values: list[int], r_values: list[int], alpha: float, beta: float, eps: float) -> tuple[np.ndarray, np.ndarray]:
    queried_coalition_ids = np.array([])
    
    candidate_coalitions = np.empty((len(k_values), len(r_values)), dtype=object)

    heavy_players = np.zeros((len(k_values), len(r_values), game.number_of_players))
    light_players = np.zeros((len(k_values), len(r_values), game.number_of_players))

    singleton_values = [game.get_value(player_to_coalition(player)) for player in range(game.number_of_players)]

    # Constructing candidate_coalitions, translate to numpy for more efficiency
    for k in range(len(k_values)):
        for r in range(len(r_values)):
            candidate_coalitions[k, r] = []

            heavy_players[k, r, :] = np.array(singleton_values) >= (
                k_values[k] * r_values[r] / sqrt(game.number_of_players)
            )
            light_players[k, r] = 1 - heavy_players[k, r]

            remaining_players = np.arange(game.number_of_players)[light_players[k, r].astype(bool)] # TODO: funguje toto tak, jak chci?
            remaining_coalition = Coalition.from_players(remaining_players)

            coalition, new_queried_coalition_ids = max_subroutine(game, remaining_coalition, k_values[k], eps)
            queried_coalition_ids = np.concatenate([queried_coalition_ids, new_queried_coalition_ids])

            while game.get_value(coalition) >= k_values[k] * r_values[r] / (2 * alpha):

                additive_vector, new_queried_coalition_ids = approx_xos_subroutine(game, coalition)
                queried_coalition_ids = np.concatenate([queried_coalition_ids, new_queried_coalition_ids])
                subcoalition = coalition

                for player in coalition.players:
                    if not additive_vector[player] >= r_values[r] / (4 * alpha * beta):
                        subcoalition -= player_to_coalition(player)
                candidate_coalitions[k, r].append(subcoalition)

                coalition, new_queried_coalition_ids = max_subroutine(game, coalition - subcoalition, k_values[k], eps)
                queried_coalition_ids = np.concatenate([queried_coalition_ids, new_queried_coalition_ids])
    
    return np.array(candidate_coalitions), np.unique(queried_coalition_ids)


def max_subroutine(game: IncompleteCooperativeGame, coalition: Coalition, size: int, eps: float) -> tuple[Coalition, np.ndarray]:
    """Computes approximate solution of k-bounded max problem for set function restricted to coalition."""
    queried_coalition_ids = []

    if coalition == Coalition(0):
        return Coalition(0), np.array([])
    else:

        singleton_values = [game.get_value(player_to_coalition(player)) for player in coalition.players]
        initial_limit = np.max(singleton_values)

        constructed_coalition = Coalition(0)
        reduced_limit = initial_limit
        while reduced_limit >= eps * initial_limit / game.number_of_players:
            for player in (coalition - constructed_coalition).players:
                if len(constructed_coalition) + 1 < size:
                    queried_coalition_ids.append((constructed_coalition + player).id)
                    if game.get_value(constructed_coalition + player) - game.get_value(constructed_coalition) >= reduced_limit:
                        constructed_coalition += player
                else:
                    return constructed_coalition, queried_coalition_ids

            reduced_limit = reduced_limit * (1 - eps)

        return constructed_coalition, queried_coalition_ids

def approx_xos_subroutine(game: IncompleteCooperativeGame, coalition: Coalition) -> tuple[np.ndarray, np.ndarray]:
    """Computes beta-XOS clause for coalition with respect to valuation of the game."""

    additive_vector = np.zeros(game.number_of_players)
    queried_coalition_ids = []

    subcoalition = Coalition(0)
    extended_subcoalition = Coalition(0)

    for player in coalition.players:
        subcoalition = extended_subcoalition
        extended_subcoalition += player
        queried_coalition_ids.append(extended_subcoalition.id)

        additive_vector[player] = (game.get_value(extended_subcoalition) - game.get_value(subcoalition))

    return additive_vector, np.array(queried_coalition_ids)

def compute_approximation(game: IncompleteCooperativeGame, candidate_coalitions: np.ndarray, k_values: list[int], r_values: list[int], alpha: float, beta: float) -> np.ndarray:
    # Constructing the approximation based on candidate_coalitions
    approximated_values = np.zeros(2**game.number_of_players)

    singleton_values = np.array([game.get_value(player_to_coalition(player)) for player in range(game.number_of_players)])

    for coalition in all_coalitions(game.number_of_players):
        if coalition.id != 0:
            max_singleton = np.max(singleton_values[list(coalition.players)])

            max_value = max_singleton
            for k in range(len(k_values)):
                for r in range(len(r_values)):
                    for candidate_coalition in candidate_coalitions[k, r]:
                        new_value = (
                            len(candidate_coalition & coalition) * r
                            / (4 * alpha * beta)
                        )
                        if new_value > max_value:
                            max_value = new_value
            approximated_values[coalition.id] = max_value
    return approximated_values

def compute_multiplicative_factor(game: IncompleteCooperativeGame, approximated_values: np.ndarray) -> float:
    """Computes the multiplicative factor of the approximation. It is v(S) / aprox(S) <= alpha, as opposed to the other algorithm."""
    multiplicative_factor = 0
    for coalition in all_coalitions(game.number_of_players):
        multiplicative_factor = max(
            multiplicative_factor,
            game.get_value(coalition) / approximated_values[coalition.id],
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
