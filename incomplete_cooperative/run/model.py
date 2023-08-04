"""Wrapper around a PPO model."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from gym import Env  # type: ignore
from sb3_contrib import MaskablePPO  # type: ignore
from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
from stable_baselines3.common.env_util import make_vec_env  # type: ignore
from stable_baselines3.common.vec_env import SubprocVecEnv  # type: ignore

from ..bounds import BOUNDS
from ..coalitions import minimal_game_coalitions
from ..game import IncompleteCooperativeGame
from ..generators import GENERATORS
from ..icg_gym import ICG_Gym
from ..random_player import RandomPolicy


def _env_generator(instance: ModelInstance) -> Env:  # pragma: nocover
    """Generate environment."""
    bounds_computer = BOUNDS[instance.game_class]
    game_generator = GENERATORS[instance.game_generator]

    incomplete_game = IncompleteCooperativeGame(instance.number_of_players, bounds_computer)
    env = ICG_Gym(incomplete_game,
                  partial(game_generator, instance.number_of_players),
                  minimal_game_coalitions(incomplete_game))

    env = ActionMasker(env, ICG_Gym.valid_action_mask)
    return env


@dataclass
class ModelInstance:
    """Model instance class."""

    name: str = "icg"
    number_of_players: int = 5
    game_class: str = "superadditive"
    game_generator: str = "factory"
    steps_per_update: int = 2048
    parallel_environments: int = 2
    random: bool = False
    run_steps_limit: int | None = None
    model_dir: Path = Path(".")

    @classmethod
    def from_parsed_arguments(cls, args) -> ModelInstance:
        """Create an instance from parsed arguments."""
        return cls(args.model_name, args.number_of_players,
                   args.game_class, args.game_generator,
                   args.steps_per_update, args.parallel_environments,
                   args.random_player, args.run_steps_limit,
                   Path(args.model_dir))

    def env_generator(self, vec_class=SubprocVecEnv) -> Env:
        """Create parallel environments."""
        return make_vec_env(_env_generator, vec_env_cls=vec_class,
                            n_envs=self.parallel_environments,
                            env_kwargs={"instance": self})

    @property
    def model_name(self) -> str:
        """Get the model path."""
        random = "_random" if self.random else ""
        return f"{self.name}_{self.game_generator}_{self.game_class}_{self.number_of_players}{random}"

    @property
    def model_path(self) -> Path:
        """Get the model path."""
        return self.model_dir / f"{self.model_name}"

    @property
    def model_out_path(self) -> Path:
        """Get the output path of the model -- with datetime."""
        return self.model_dir / f"{self.model_name}_{datetime.now().isoformat()}"

    @property
    def model(self) -> MaskablePPO:
        """Get model."""
        envs = self.env_generator()
        if self.random:
            return MaskablePPO(RandomPolicy, envs, n_steps=self.steps_per_update, verbose=10)
        return MaskablePPO.load(self.model_path, envs) if self.model_path.with_suffix(".zip").exists() \
            else MaskablePPO("MlpPolicy", envs, n_steps=self.steps_per_update, verbose=10)

    def save(self, model: MaskablePPO) -> None:
        """Save model."""
        model.save(self.model_path)


def add_model_arguments(ap) -> None:
    """Add arguments specific for the model instance."""
    defaults = ModelInstance()
    ap.add_argument("--model-name", required=False, default=defaults.name,
                    help="The name of the model.")
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
    ap.add_argument("--model-dir", type=str, default=".")
