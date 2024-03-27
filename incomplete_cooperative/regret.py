"""This module implements regret minimization for the Principal's Problem."""
from itertools import chain, combinations
from typing import Iterable

import numpy as np  # type: ignore
import scipy  # type: ignore

from incomplete_cooperative.coalitions import Coalition, all_coalitions


def metacoalition_ids_by_coalition_size(number_of_players: int, limit_of_revealed: int) -> np.ndarray:
    """Get a list of viable meta-coalition ids sorted by the size of the meta-coalition.

    Arguments:
        number_of_players: number of players in the game.
        limit_of_revealed: maximum size of the meta-coalition.
    """
    # number of coalitions, excluding those in the K_0 set.
    number_of_coalitions = 2**number_of_players - number_of_players - 2
    limit_of_revealed = min(number_of_coalitions, limit_of_revealed)

    coalition_ids = range(number_of_coalitions)
    all_metacoalitions = chain.from_iterable(
        combinations(coalition_ids, size) for size in range(limit_of_revealed + 1)
    )
    all_metacoalition_ids = map(lambda x: Coalition.from_players(x).id, all_metacoalitions)
    return np.fromiter(
        all_metacoalition_ids, dtype=int,
        # we can calculate the size of the resulting array using sum of the binomial coefficients for each size
        count=coalitions_up_to(number_of_coalitions, limit_of_revealed))


def coalitions_up_to(number_of_players: int, size_limit: int) -> int:
    """Calculate the number of coalitions _smaller or equal_ to `size_limit`."""
    return int(scipy.special.comb(number_of_players, np.arange(size_limit + 1)).sum())


def get_coalition_player_id_map(number_of_players: int) -> np.ndarray:
    """Map a coalition to its 'player_id'."""
    viable_coalitions = [x.id for x in all_coalitions(number_of_players)
                         if len(x) not in [0, 1, number_of_players]]
    ret = np.zeros(2**number_of_players, dtype=int) - 1
    ret[viable_coalitions] = range(len(viable_coalitions))
    return ret


RMValue = np.float32


class GameRegretMinimizer:
    """The main regret minimizer object, holding the regret information.

    We use the notion of "meta-coalitions" to refer to sets of coalitions.
    This means that each coalition can be viewed as a player in a new game.
    It would be wasteful to consider _all_ coalitions as players --
    we don't ever need to consider coalitions which are already in the K_0 set.

    That said, because we don't use all coalitions, we need to re-map their ids
    (without that, we would have gaps in the ids, leading to inefficiencies).
    To achieve this, we use a map from the coalitions (excluding K_0) to their "player_ids".

    We minimize regret in every internal node of the game tree.
    When updating regret, we need to scan through the meta-coalitions from largest to smallest.
    This identification is refered to as the "rank" of the meta-coalition.
    Such an order is found using the `metacoalition_ids_by_coalition_size` function, and stored
    in `meta_id_to_rank`.

    Attributes:
        number_of_players: number of players in the games.
        number_of_coalitions: number of coalitions which aren't in K_0.
        limit_of_revealed: maximum number of coalitions to reveal, i.e. maximum size of metacoal.
        meta_rank_to_id: map metacoalition size rank -> metacoalition id.
        meta_id_to_rank: map metacoalition id -> metacoalition size rank.
        cumulative_regret: an array of instant regrets of each regret minimizer (zero if an action is inapplicable).
            This array has size (number_of_regret_minimizers, number_of_coalitions).
            It is indexed by the meta-coalition ranks.
    """

    def __init__(self, number_of_players: int, limit_of_revealed: int, plus: bool = False) -> None:
        """Initialize RM.

        Arguments:
            number_of_players: number of players in the games.
            limit_of_revealed: maximum number of coalitions to reveal.
            plus: whether to compute regret matching plus. Default: False.
        """
        self.plus = plus
        self.number_of_players = number_of_players
        # number of coalitions we are interested in -- ie not those in K_0
        self.number_of_coalitions = 2 ** number_of_players - number_of_players - 2
        self.limit_of_revealed = limit_of_revealed

        # map metacoalition size rank -> metacoalition id
        self.meta_rank_to_id = metacoalition_ids_by_coalition_size(number_of_players, limit_of_revealed)
        # these are the metacoalitions of size <= limit_of_revealed
        self.viable_metacoalitions = len(self.meta_rank_to_id)
        # map metacoalition id -> metacoalition size rank
        self.meta_id_to_rank = np.zeros(self.viable_metacoalitions, dtype=int)
        self.meta_id_to_rank[self.meta_rank_to_id] = np.arange(self.viable_metacoalitions)

        # regret minimizer for every internal node -- except for the bottom layer of game tree
        self.number_of_regret_minimizers = coalitions_up_to(self.number_of_coalitions, limit_of_revealed - 1)

        self.coalitions_to_player_ids = get_coalition_player_id_map(number_of_players)
        self.cumulative_regret = np.zeros((self.number_of_regret_minimizers, self.number_of_coalitions), dtype=RMValue)
        self.cumulative_strategy = np.zeros((self.number_of_regret_minimizers, self.number_of_coalitions),
                                            dtype=RMValue)
        self.iteration = 0

    def get_metacoalition_id(self, metacoalition: Iterable[Coalition]) -> int:
        """Properly get an id of a metacoalition."""
        coalition_player_ids = self.coalitions_to_player_ids[
            [x.id for x in metacoalition if len(x) not in [0, 1, 2**self.number_of_players]]]
        return np.sum(2 ** coalition_player_ids)

    def regret_matching_strategy(self, past_actions: Iterable[Coalition] | int) -> np.ndarray:
        """Get the action predicted by the strategy using the Regret Matching minimizer.

        If past_actions is an int, it is considered a metacoalition id.
        Otherwise, it is considered as a list of coalitions of the original game.

        This returns the current prediction, not the averaged strategy over time.
        """
        metacoalition = past_actions if isinstance(past_actions, int) else self.get_metacoalition_id(past_actions)
        rank = self.meta_id_to_rank[metacoalition]
        positive_regret = self.cumulative_regret[rank] * (self.cumulative_regret[rank] > 0)
        if positive_regret.sum() == 0:
            positive_regret = np.ones(self.number_of_coalitions)
            used_coalitions = list(Coalition(metacoalition).players)
            positive_regret[used_coalitions] = 0
        return positive_regret / positive_regret.sum()

    def get_average_strategy(self, past_actions: Iterable[Coalition]) -> np.ndarray:
        """Get average strategy, in terms of actual coalitions in the original game."""
        metacoalition = int(self.get_metacoalition_id(past_actions))
        rank = self.meta_id_to_rank[metacoalition]
        # in `regret minimization` we only add the "previous" regret matching strategy, not the current one,
        # so we need to add it here
        average_strategy = self.cumulative_strategy[rank] + self.regret_matching_strategy(metacoalition)
        average_strategy = average_strategy / average_strategy.sum()
        # translate from player_ids to coalition ids
        # index by the original player ids, or -1. Then, remove all the positions where it was -1,
        # leaving the original values, just reordered, and zeros where missing
        return average_strategy[self.coalitions_to_player_ids] * (self.coalitions_to_player_ids > -0.5)

    def regret_min_iteration(self, terminal_losses: np.ndarray, used_actions: list[list[Coalition]]) -> None:
        """Scan every possible action sequence, and update the regret of all the regret minimizers.

        Arguments:
            terminal_losses: the loss of the bottom layer of the game tree.
                    this should be an array of size (number_of_coalitions,).
            used_actions: the list of list of actions used to generate the bottom_losses.
        """
        self.iteration += 1
        used_metacoalition_ranks = [self.meta_id_to_rank[self.get_metacoalition_id(x)]
                                    for x in used_actions]
        q_values = np.zeros((self.number_of_regret_minimizers, self.number_of_coalitions),
                            dtype=RMValue)
        experienced_losses = np.zeros(self.viable_metacoalitions, dtype=RMValue)
        experienced_losses[used_metacoalition_ranks] = terminal_losses
        for i in range(self.number_of_regret_minimizers - 1, -1, -1):
            metacoalition = self.meta_rank_to_id[i]
            metacoalition_coal = Coalition(metacoalition)
            next_coalition_pids = list(metacoalition_coal.inverted(self.number_of_coalitions).players)
            next_metacoalitions = [(metacoalition_coal + x).id for x in next_coalition_pids]
            next_metacoalition_ranks = self.meta_id_to_rank[next_metacoalitions]
            q_values[i, next_coalition_pids] = experienced_losses[next_metacoalition_ranks]
            regret_matching_strategy = self.regret_matching_strategy(int(metacoalition))
            experienced_losses[i] = (q_values[i] * regret_matching_strategy).sum()
            weight = self.iteration if self.plus else 1
            self.cumulative_strategy[i] += weight * regret_matching_strategy

        self.cumulative_regret += q_values - experienced_losses[np.arange(self.number_of_regret_minimizers), None]
        if self.plus:
            self.cumulative_regret *= self.cumulative_regret > 0
