import gym
from typing import Callable
from itertools import chain, combinations
import numpy as np
import math
from .game import IncompleteCooperativeGame, CoalitionPlayers, Value, Coalition

ValuesGenerator = Callable[[IncompleteCooperativeGame, Coalition], Value]


class ICG_Gym(gym.Env):
    """A `gym` for incomplete cooperative games."""

    def __init__(self, game: IncompleteCooperativeGame,
                 full_game: IncompleteCooperativeGame,
                 initially_known_values: list[CoalitionPlayers],
                 new_values_generator: ValuesGenerator) -> None:
        """Initialize gym."""
        super().__init__()
        self.game = game
        # TODO: normalize game.
        self.full_game = full_game
        self.initial_values = map(full_game.get_value, map(full_game.players_to_coalition, initially_known_values))
        self.game.set_known_values(initially_known_values)
        self.explorable_coalitions = 2**self.game.number_of_players - self.game.number_of_players - 1

        if self.game_type == 'superadditive':  # TODO: implement later.
            for p in self.game.coalitions:
                if len(p) > 1:  # Ignore singletons
                    proper_subset = list(chain.from_iterable(combinations(p, r) for r in range(1, len(p))))
                    for i in range(len(proper_subset) // 2):  # All non-doubled splits of the coalition
                        a, b = tuple(chain(proper_subset[i])), tuple([e for e in p if e not in proper_subset[i]])
                        # Superadditivity constraint
                        self.nodes[p][1] = max(self.nodes[p][1], self.nodes[a][1] + self.nodes[b][1])
                    # Now add a bit of noise
                    self.nodes[p][1] += 1  # Modify the true value by (later random) constant

            # Reveal the value of the grand coalition
            self.nodes[self.powerset[-1]][0] = 1
            self.nodes[self.powerset[-1]][1:] = self.nodes[self.powerset[-1]][1]

            # Normalize values to [0, 1]
            grand_coalition_value = self.nodes[self.powerset[-1]][1]
            for p in self.powerset:
                self.nodes[p][1:] /= grand_coalition_value

            self.compute_bounds()

        else:
            raise NotImplementedError()

        self.initial_nodes = {}
        for s, v in self.nodes.items():
            self.initial_nodes[s] = v.copy()

        self.observation_space = gym.spaces.Box(low=np.zeros(self.explorable_coalitions),
                                                high=np.ones(self.explorable_coalitions),
                                                dtype=np.float32)

        self.action_space = gym.spaces.Discrete(self.explorable_coalitions)

    def compute_bounds(self):
        # First, propagate the lower bounds up the tree
        for p in self.powerset:
            if len(p) > 1:  # Ignore singletons, since they are given
                proper_subset = list(chain.from_iterable(combinations(p, r) for r in range(1, len(p))))
                for i in range(len(proper_subset) // 2):  # All non-doubled splits of the coalition
                    a, b = tuple(chain(proper_subset[i])), tuple([e for e in p if e not in proper_subset[i]])
                    # Lower bound is the maximum over subsets, sum of their lower bounds if not revealed
                    if self.nodes[p][0]:
                        self.nodes[p][2] = self.nodes[p][1]
                    else:
                        self.nodes[p][2] = max(self.nodes[p][2], self.nodes[a][2] + self.nodes[b][2])

        # Second, given upper bound of a (or b) is upper bound of p=a+b minus lower bound of b (or a)
        for p in reversed(self.powerset):
            if len(p) > 1:  # Ignore singletons, since they are given
                proper_subset = list(chain.from_iterable(combinations(p, r) for r in range(1, len(p))))
                for i in range(len(proper_subset) // 2):  # All non-doubled splits of the coalition
                    a, b = tuple(chain(proper_subset[i])), tuple([e for e in p if e not in proper_subset[i]])
                    if self.nodes[a][0]:
                        self.nodes[a][3] = self.nodes[a][1]
                    else:
                        self.nodes[a][3] = max(self.nodes[a][3], self.nodes[p][3] - self.nodes[b][2])

                    if self.nodes[b][0]:
                        self.nodes[b][3] = self.nodes[b][1]
                    else:
                        self.nodes[b][3] = max(self.nodes[b][3], self.nodes[p][3] - self.nodes[a][2])

    def _exploitability(self):
        n = self.num_players
        true_shapley = np.zeros(n)
        max_shapley = np.zeros(n)
        for p in range(n):
            for s in self.powerset:
                if not p in s:
                    s_revealed, s_true, s_min, _ = self.nodes[s]  # Values of the set without P
                    # Set s + p
                    sp = list(s) + [p, ]
                    sp.sort()
                    sp = tuple(sp)
                    sp_revealed, sp_true, _, sp_max = self.nodes[sp]   # Values when P is included

                    # Factorial term in front of the value difference
                    prefactor = math.factorial(len(s)) * math.factorial(self.num_players - len(s) - 1)
                    true_shapley[p] += prefactor * (sp_true - s_true)
                    _max = sp_max * (1 - sp_revealed) + sp_true * sp_revealed
                    _min = s_min * (1 - s_revealed) + s_true * s_revealed
                    max_shapley[p] += prefactor * (sp_max - s_min)

        # Divide by factorial
        true_shapley /= math.factorial(self.num_players)
        max_shapley /= math.factorial(self.num_players)

        return np.sum(max_shapley - true_shapley)

    def valid_action_mask(self):
        mask = np.array([self.nodes[s][0] for s in self.explorable_coalitions])
        return mask

    def reset(self):
        # Mark all nodes as unknown except for the singletons and grand coalition
        for s in self.powerset:
            self.nodes[s][:] = self.initial_nodes[s].copy()

        self.compute_bounds()

        # TODO: Later we want to modify the true values

        # Return initial state
        mask = np.array([self.nodes[s][0] for s in self.explorable_coalitions])
        true_state = np.array([self.nodes[s][1] for s in self.explorable_coalitions])

        return mask * true_state

    def step(self, action: int):
        '''
        Implementing one step of the arbitor, revealing coalition and computing exploitability

        :param action: Int choosing which coalition to reveal
        :return: next_state: np.array of values of revealed coalitions, zero otherwise
                         reward: negative exploitability
                         done: bool if all coalitions are revealed
                         info: empty list (no additional info)
        '''
        # The chosen coalition for revealing, skipping the singletons
        chosen_coalition = self.powerset[action + self.num_players]
        self.nodes[chosen_coalition][0] = 1
        self.compute_bounds()

        mask = np.array([self.nodes[s][0] for s in self.explorable_coalitions])
        true_state = np.array([self.nodes[s][1] for s in self.explorable_coalitions])

        masked_state = mask * true_state

        # Reward is negative exploitability
        reward = - self._exploitability()

        # The game is done if the agent reveals all coalitions
        done = np.all(mask == 1)

        return masked_state, reward, done, {}
