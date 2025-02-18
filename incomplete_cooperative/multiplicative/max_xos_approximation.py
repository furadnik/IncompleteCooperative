"""Approximate an incomplete game using Max XOS approximation."""
from math import sqrt

import numpy as np

from ..coalitions import Coalition, all_coalitions, player_to_coalition
from ..game import IncompleteCooperativeGame
from ..protocols import Game


def compute_max_xos_approximation(game: Game, alpha: float = 3.7844223824,
                                  beta: float = 1, eps: float = 0.05) -> tuple[np.ndarray, IncompleteCooperativeGame]:
    """Compute the Max XOS approximation.

    The game is expected to be subadditive and monotone increasing.

    Return the array of ids of queried coalitions, along with the approximated game.
    """
    k_values, r_values = _get_k_r_values(game.number_of_players)

    candidate_coalitions, queried_coalition_ids = _compute_candidate_coalitions_and_query_values(
        game, k_values, r_values, alpha, beta, eps
    )

    approximated_values = _compute_approximation(game, candidate_coalitions, k_values, r_values, alpha, beta)
    approximated_game = IncompleteCooperativeGame(game.number_of_players)
    approximated_game.set_values(approximated_values)

    return queried_coalition_ids, approximated_game


def _get_k_r_values(number_of_players: int) -> tuple[list[int], list[int]]:
    """Compute parameters of the Max XOS apx algorithm."""
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


def _compute_candidate_coalitions_and_query_values(
    game: Game, k_values: list[int], r_values: list[int], alpha: float, beta: float, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    queried_coalition_ids = np.array([])

    candidate_coalitions = np.empty((len(k_values), len(r_values)), dtype=object)

    heavy_players = np.zeros((len(k_values), len(r_values), game.number_of_players))
    light_players = np.zeros((len(k_values), len(r_values), game.number_of_players))

    singleton_values = np.array([game.get_value(player_to_coalition(player)) for player in range(game.number_of_players)])
    assert np.all(singleton_values >= 1)

    # Constructing candidate_coalitions, translate to numpy for more efficiency
    for k in range(len(k_values)):
        for r in range(len(r_values)):
            candidate_coalitions[k, r] = []

            heavy_players[k, r, :] = singleton_values >= (
                k_values[k] * r_values[r] / sqrt(game.number_of_players)
            )
            light_players[k, r] = 1 - heavy_players[k, r]

            remaining_players = np.arange(game.number_of_players)[light_players[k, r].astype(bool)]
            remaining_coalition = Coalition.from_players(remaining_players)

            coalition, new_queried_coalition_ids = _max_subroutine(game, remaining_coalition, k_values[k], eps)
            queried_coalition_ids = np.concatenate([queried_coalition_ids, new_queried_coalition_ids])

            while game.get_value(coalition) >= k_values[k] * r_values[r] / (2 * alpha):

                additive_vector, new_queried_coalition_ids = _approx_xos_subroutine(game, coalition)
                queried_coalition_ids = np.concatenate([queried_coalition_ids, new_queried_coalition_ids])
                subcoalition = coalition

                for player in coalition.players:
                    if not additive_vector[player] >= r_values[r] / (4 * alpha * beta):
                        subcoalition -= player_to_coalition(player)
                candidate_coalitions[k, r].append(subcoalition)

                coalition, new_queried_coalition_ids = _max_subroutine(game, coalition - subcoalition, k_values[k], eps)
                queried_coalition_ids = np.concatenate([queried_coalition_ids, new_queried_coalition_ids])

    return np.array(candidate_coalitions), np.unique(queried_coalition_ids)


def _max_subroutine(game: Game, coalition: Coalition,
                    size: int, eps: float) -> tuple[Coalition, np.ndarray]:
    """Compute approximate solution of k-bounded max problem for set function restricted to coalition."""
    queried_coalition_ids: list[int] = []

    if coalition == Coalition(0):
        return Coalition(0), np.array([])
    else:

        singleton_values = np.array([game.get_value(player_to_coalition(player)) for player in coalition.players])
        assert np.all(singleton_values >= 1)
        initial_limit = np.max(singleton_values)

        constructed_coalition = Coalition(0)
        reduced_limit = initial_limit
        while reduced_limit >= eps * initial_limit / game.number_of_players:
            for player in (coalition - constructed_coalition).players:
                if len(constructed_coalition) + 1 >= size:
                    return constructed_coalition, np.array(queried_coalition_ids)
                queried_coalition_ids.append((constructed_coalition + player).id)
                player_contrib = game.get_value(constructed_coalition + player) - game.get_value(constructed_coalition)
                if player_contrib >= reduced_limit:
                    constructed_coalition += player

            reduced_limit = reduced_limit * (1 - eps)

        return constructed_coalition, np.array(queried_coalition_ids)


def _approx_xos_subroutine(game: Game, coalition: Coalition) -> tuple[np.ndarray, np.ndarray]:
    """Compute beta-XOS clause for coalition with respect to valuation of the game."""
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


def _compute_approximation(game: Game, candidate_coalitions: np.ndarray, k_values: list[int],
                           r_values: list[int], alpha: float, beta: float) -> np.ndarray:
    """Construct the approximation based on candidate_coalitions."""
    approximated_values = np.zeros(2**game.number_of_players)
    singleton_values = np.array([game.get_value(player_to_coalition(player))
                                 for player in range(game.number_of_players)])
    assert np.all(singleton_values >= 1)

    for coalition in all_coalitions(game.number_of_players):
        if coalition.id != 0:
            max_singleton = np.max(singleton_values[list(coalition.players)])

            max_value = max_singleton
            for k in range(len(k_values)):
                for r in range(len(r_values)):
                    for candidate_coalition in candidate_coalitions[k, r]:
                        new_value = len(candidate_coalition & coalition) * r / (4 * alpha * beta)
                        if new_value > max_value:  # pragma: nocover
                            max_value = new_value

            approximated_values[coalition.id] = max_value
    return approximated_values
