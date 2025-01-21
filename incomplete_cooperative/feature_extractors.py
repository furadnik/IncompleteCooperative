"""Define custom feature extractors."""
import torch
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """A custom feature extractor.

    - Flattens the input
    - Applies LayerNorm
    - Applies a linear layer + activation
    """

    def __init__(self, observation_space: Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )

        self._features_dim = 64

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from raw observations."""
        return self.net(observations)


EXTRACTORS = {
    "ln_64": CustomFeatureExtractor
}
