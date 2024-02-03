"""Provide a custom model to the PPO, so it scales better with the number of players.

https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
"""
import torch  # type: ignore
from gymnasium import spaces  # type: ignore
from stable_baselines3.common.torch_layers import \
    BaseFeaturesExtractor  # type: ignore

from .coalitions import all_coalitions


def get_action_encoding(number_of_players: int) -> torch.Tensor:
    """Return a torch tensor with encoding of every possible action."""
    encoding = torch.zeros((2 ** number_of_players - number_of_players - 2, number_of_players))
    for i, coalition in enumerate(
        x for x in all_coalitions(number_of_players) if len(x) not in [0, 1, number_of_players]
    ):
        encoding[i, list(coalition.players)] = 1

    return encoding


class CustomPerActionPolicyNet(BaseFeaturesExtractor):
    """Custom policy network, sharing neurons for all actions.

    Currently only supports the default explorable coalitions: all but the minimal information.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, number_of_players: int = -1):
        """Compute the encoding and store the initial layers.

        The size of the input encoding is 2^n - n - 2, where n is the number of players.
        """
        super().__init__(observation_space, features_dim)
        if number_of_players < 3:
            raise ValueError("This works with at least three players.")
        self.encoding = get_action_encoding(number_of_players)
        self.linear_1 = torch.nn.Linear(self.encoding.shape[0] + self.encoding.shape[1], features_dim)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(features_dim, 1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward the network.

        As per the issue text, we want to first consider the input as a batch of observations,
        having as many as actions. We then concatenate the encoding of each action to the input.
        Then pass it to the network to operate on the concatenated input.
        Finally, run softmax over it.
        """
        num_actions = self.encoding.shape[0]
        per_action_observations = observations.repeat(num_actions, 1)
        per_action_observations = torch.cat([per_action_observations, self.encoding], dim=-1)
        x = self.linear_1(per_action_observations)
        x = self.relu(x)
        x = self.linear_2(x)  # At this point we have a tensor of shape (num_actions, 1)
        x = self.softmax(x[:, 0])  # Here we forget that the 1 is there.
        return x
