"""Wrapper around a PPO model."""
from __future__ import annotations

from datetime import datetime
from argparse import Namespace
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Callable, cast

import numpy as np
from gymnasium import Env  # type: ignore
from sb3_contrib import MaskablePPO  # type: ignore
from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
from stable_baselines3.common.env_util import make_vec_env  # type: ignore
from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
from stable_baselines3.common.vec_env import SubprocVecEnv  # type: ignore
from stable_baselines3.common.vec_env import VecEnv

from ..bounds import BOUNDS
from ..coalitions import minimal_game_coalitions
from ..game import IncompleteCooperativeGame
from ..generators import GENERATORS
from ..icg_gym import ICG_Gym
from ..random_player import RandomPolicy

ENVIRONMENTS: dict[str, type[SubprocVecEnv] | type[DummyVecEnv]] = {
    "parallel": SubprocVecEnv,
    "sequential": DummyVecEnv
}


def _env_generator(instance: ModelInstance) -> Env:  # pragma: nocover
    """Generate environment."""
    bounds_computer = BOUNDS[instance.game_class]
    game_generator = GENERATORS[instance.game_generator]

    incomplete_game = IncompleteCooperativeGame(instance.number_of_players, bounds_computer)
    env = ICG_Gym(incomplete_game,
                  partial(game_generator, instance.number_of_players),
                  minimal_game_coalitions(incomplete_game),
                  done_after_n_actions=instance.run_steps_limit)

    return ActionMasker(env, cast(Callable[[Env], np.ndarray], ICG_Gym.valid_action_mask))


@dataclass
class ModelInstance:
    """Model instance class."""

    number_of_players: int = 5
    game_class: str = "superadditive"
    game_generator: str = "factory"
    steps_per_update: int = 2048
    parallel_environments: int = 2
    random_player: bool = False
    run_steps_limit: int | None = None
    model_dir: Path = Path(".")
    model_path: Path = None  # type: ignore[assignment]
    unique_name: str = str(datetime.now().isoformat())
    environment: str = "sequential"

    def __post_init__(self) -> None:
        """Exit model path."""
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)
        if self.model_path is None:
            self.model_path = self.model_dir / "model"

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
        return make_vec_env(_env_generator, vec_env_cls=self.environment_class,
                            n_envs=self.parallel_environments,
                            env_kwargs={"instance": self})

    @property
    def model(self) -> MaskablePPO:
        """Get model."""
        envs = self.env_generator()
        if self.random_player:
            return MaskablePPO(RandomPolicy, envs, n_steps=self.steps_per_update, verbose=10)
        return MaskablePPO.load(self.model_path, envs) if self.model_path.with_suffix(".zip").exists() \
            else MaskablePPO("MlpPolicy", envs, n_steps=self.steps_per_update, verbose=10)

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
    ap.add_argument("--random-player", action="store_true")
    ap.add_argument("--model-dir", type=Path, default=defaults.model_dir)
    ap.add_argument("--model-path", type=Path, required=False)
    ap.add_argument("--unique-name", type=str, required=False, default=defaults.unique_name)
    ap.add_argument("--environment", type=str, required=False, default=defaults.environment)
