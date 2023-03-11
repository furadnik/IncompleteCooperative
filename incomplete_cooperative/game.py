import gym
from itertools import chain, combinations
import numpy as np
import math


class Incomplete_Cooperative_Game(gym.Env):
    def __init__(self):
        super(Incomplete_Cooperative_Game, self).__init__()

        self.num_players = 3  # args.num_players
        self.game_type = 'superadditive'  # Other types will be supported later
        self.singleton_values = [1, 2, 3]  # args.singleton_values

        # Create the set of all subsets of players (omitting the empty set, since its values is fixed)
        player_list = [i for i in range(self.num_players)]
        self.powerset = list(chain.from_iterable(combinations(player_list, r) for r in range(1, self.num_players + 1)))
        self.explorable_coalitions = self.powerset[self.num_players:-1]

        # For each element of the powerset, we will have:
        # (0) if it is known,
        # (1) the true (initially unknown) value,
        # (2) lower and (3) upper bound.
        self.nodes = {p: np.zeros(4) for p in self.powerset}
        # Populate singletons
        for player, singleton in enumerate(self.singleton_values):
            self.nodes[(player, )][0] = 1  # The values of singletons are initially known
            self.nodes[(player, )][1:] = self.singleton_values[player]  # Since they are known, the bounds are tight

        if self.game_type == 'superadditive':
            for p in self.powerset:
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

        self.observation_space = gym.spaces.Box(low=np.zeros(len(self.explorable_coalitions)),
                                                                                        high=np.ones(len(self.explorable_coalitions)),
                                                                                        dtype=np.float32)

        self.action_space = gym.spaces.Discrete(len(self.explorable_coalitions))

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
