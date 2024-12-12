"""Wrapper around a PPO model."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from gymnasium import Env  # type: ignore
from sb3_contrib import MaskablePPO  # type: ignore
from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
from stable_baselines3.common.env_util import make_vec_env  # type: ignore
from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
from stable_baselines3.common.vec_env import SubprocVecEnv  # type: ignore
from stable_baselines3.common.vec_env import VecEnv
from torch.nn.modules.activation import ReLU, Tanh  # type: ignore

from ..bounds import BOUNDS
from ..coalitions import minimal_game_coalitions
from ..exploitability import compute_exploitability
from ..game import IncompleteCooperativeGame
from ..generators import GENERATORS
from ..icg_gym import ICG_Gym
from ..normalize import NormalizableGame
from ..norms import l1_norm, l2_norm, linf_norm
from ..protocols import GapFunction, IncompleteGame, Value

ENVIRONMENTS: dict[str, type[SubprocVecEnv] | type[DummyVecEnv]] = {
    "parallel": SubprocVecEnv,
    "sequential": DummyVecEnv
}

ACTIVATION_FNS = {
    "relu": ReLU,
    "tanh": Tanh
}

GAP_FUNCTIONS: dict[str, Callable[[IncompleteGame], Value]] = {
    "exploitability": compute_exploitability,
    "l1_norm": l1_norm,
    "l2_norm": l2_norm,
    "linf_norm": linf_norm
}


@dataclass
class ModelInstance:
    """Model instance class."""

    number_of_players: int = 5
    game_class: str = "superadditive"
    game_generator: str = "factory"
    gap_function: str = "exploitability"
    steps_per_update: int = 2048
    parallel_environments: int = 2
    run_steps_limit: int | None = None
    model_dir: Path = Path(".")
    model_path: Path = None  # type: ignore[assignment]
    unique_name: str = str(datetime.now().isoformat())
    environment: str = "sequential"
    policy_activation_fn: str = "relu"
    gamma: float = 1
    ent_coef: float = 0.1
    seed: int = int(datetime.now().timestamp() * 1000)

    @property
    def seed_32(self) -> int:
        """Seed mod 32."""
        return self.seed % 2**32

    def game_generator_fn(self) -> NormalizableGame:
        """Get the game generator, with a pre-set random generator."""
        return GENERATORS[self.game_generator](self.number_of_players, self.game_generator_rng)

    def __post_init__(self) -> None:
        """Exit model path."""
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)
        if self.model_path is None:
            self.model_path = self.model_dir / "model"
        self.game_generator_rng = np.random.default_rng(self.seed)

    def get_env(self) -> ICG_Gym:
        """Get env."""
        bounds_computer = BOUNDS[self.game_class]
        incomplete_game = IncompleteCooperativeGame(self.number_of_players, bounds_computer)

        return ICG_Gym(incomplete_game,
                       self.game_generator_fn,
                       minimal_game_coalitions(incomplete_game),
                       self.gap_function_callable,
                       done_after_n_actions=self.run_steps_limit)

    @property
    def gap_function_callable(self) -> GapFunction:
        """Gap function."""
        return GAP_FUNCTIONS[self.gap_function]

    @classmethod
    def from_parsed_arguments(cls, args: Namespace) -> ModelInstance:
        """Create an instance from parsed arguments."""
        field_names = [x.name for x in fields(cls)]
        return cls(**{key: value for key, value in vars(args).items() if key in field_names})

    @property
    def environment_class(self) -> type[DummyVecEnv] | type[SubprocVecEnv]:
        """TODO: implement later."""
        return ENVIRONMENTS.get(self.environment, DummyVecEnv)

    def env_generator(self) -> VecEnv:
        """Create parallel environments."""
        return make_vec_env(self.get_env, vec_env_cls=self.environment_class,
                            n_envs=self.parallel_environments)

    @property
    def policy_activation_fn_choice(self) -> Any:
        """Get the chosen activation function."""
        return ACTIVATION_FNS[self.policy_activation_fn]

    @property
    def model(self) -> MaskablePPO:
        """Get model."""
        envs = self.env_generator()
        return MaskablePPO.load(self.model_path, envs) if self.model_path.with_suffix(".zip").exists() \
            else MaskablePPO("MlpPolicy", envs, n_steps=self.steps_per_update, ent_coef=self.ent_coef,
                             policy_kwargs={"activation_fn": self.policy_activation_fn_choice},
                             verbose=10, gamma=self.gamma, seed=self.seed_32)

    def save(self, model: MaskablePPO) -> None:
        """Save model."""
        model.save(self.model_path)


def add_model_arguments(ap) -> None:
    """Add arguments specific for the model instance."""
    defaults = ModelInstance()
    ap.add_argument("--number-of-players", default=defaults.number_of_players, type=int,
                    help="Set the number of players in the game.")
    ap.add_argument("--game-class", default=defaults.game_class,
                    help="This is the class the agent thinks the game belongs to.", choices=BOUNDS.keys())
    ap.add_argument("--game-generator", default=defaults.game_generator,
                    help="Set the game generator.", choices=GENERATORS.keys())
    ap.add_argument("--steps-per-update", default=defaults.steps_per_update,
                    type=int, help="Steps in one epoch when learning.")
    ap.add_argument("--parallel-environments", default=defaults.parallel_environments, type=int)
    ap.add_argument("--run-steps-limit", default=defaults.run_steps_limit, type=int)
    ap.add_argument("--model-dir", type=Path, default=defaults.model_dir)
    ap.add_argument("--model-path", type=Path, required=False)
    ap.add_argument("--ent-coef", type=float, required=False, default=defaults.ent_coef)
    ap.add_argument("--gamma", type=float, required=False, default=defaults.gamma)
    ap.add_argument("--unique-name", type=str, required=False, default=defaults.unique_name)
    ap.add_argument("--environment", type=str, required=False, default=defaults.environment)
    ap.add_argument("--policy-activation-fn", type=str, choices=ACTIVATION_FNS.keys(), required=False,
                    default=defaults.policy_activation_fn)
    ap.add_argument("--gap-function", type=str, choices=GAP_FUNCTIONS.keys(), required=False,
                    default=defaults.gap_function)
    ap.add_argument("--seed", type=int, required=False, default=defaults.seed)
