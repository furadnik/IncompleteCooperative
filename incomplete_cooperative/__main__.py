"""CLI."""
import random
import sys
import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
from gym import Env  # type: ignore
from sb3_contrib import MaskablePPO  # type: ignore
from sb3_contrib.common.maskable.utils import get_action_masks  # type: ignore
from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
from stable_baselines3.common.env_util import make_vec_env  # type: ignore

from .bounds import BOUNDS
from .coalitions import minimal_game_coalitions
from .game import IncompleteCooperativeGame
from .generators import GENERATORS
from .icg_gym import ICG_Gym


def learn_func(env, save_path: Path, parsed_args) -> None:
    """Model learning."""
    envs = make_vec_env(env, n_envs=parsed_args.parallel_environments)
    model = MaskablePPO.load(save_path, envs) if save_path.with_suffix(".zip").exists() \
        else MaskablePPO("MlpPolicy", envs, verbose=10)

    model.learn(parsed_args.repetitions)
    model.save(save_path)


def eval_func(env, save_path: Path, parsed_args) -> None:
    """Evaluate the model."""
    model = MaskablePPO.load(save_path)
    envs = make_vec_env(env, n_envs=parsed_args.parallel_environments)
    obs = envs.reset()
    while True:  # TODO: how do we actually want to do this? Do we want the parallel envs here?
        action_masks = get_action_masks(envs)
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, rewards, dones, info = envs.step(action)
        print(rewards)
        if np.all(dones):
            break


def get_argument_parser() -> ArgumentParser:
    """TODO: implement later."""
    ap = ArgumentParser(description="Allow usage of the ICG module from the command-line.")

    ap.add_argument("--number-of-players", default=9, type=int,
                    help="Set the number of players in the game.")
    ap.add_argument("--seed", required=False,
                    help="Set the seed for possible random generation.")
    ap.add_argument("--game-type", default="superadditive",
                    help="Set the game type.")
    ap.add_argument("--game-generator", default="factory",
                    help="Set the game generator.")
    ap.add_argument("--model-path", required=False,
                    help="The path to/from which the model will be stored/loaded.")
    ap.add_argument("--parallel-environments", default=5, type=int)
    ap.add_argument("--repetitions", default=50_000, type=int)
    subparsers = ap.add_subparsers(required=True)
    learn = subparsers.add_parser("learn")
    learn.set_defaults(func=learn_func)
    eval = subparsers.add_parser("evaluate")
    eval.set_defaults(func=eval_func)
    return ap


def env_generator(parsed_args) -> Env:
    """Generate environment."""
    number_of_players = parsed_args.number_of_players
    bounds_computer = BOUNDS[parsed_args.game_type]
    game_generator = GENERATORS[parsed_args.game_generator]

    incomplete_game = IncompleteCooperativeGame(number_of_players, bounds_computer)
    env = ICG_Gym(incomplete_game,
                  partial(game_generator, number_of_players),
                  minimal_game_coalitions(incomplete_game)
                  )

    env = ActionMasker(env, ICG_Gym.valid_action_mask)
    return env


def get_model_path(parsed_args) -> Path:
    """Get the model path."""
    return Path(parsed_args.model_path or ("incomplete_cooperative_model_" + str(time.time())))


def main(ap=get_argument_parser(), args=sys.argv) -> None:
    """Run the main script."""
    args = args[1:]
    parsed_args = ap.parse_args(args)
    print(parsed_args)

    if parsed_args.seed is not None:  # does not apply to the agent, do we want to change that?
        random.seed(parsed_args.seed)

    env = partial(env_generator, parsed_args)
    save_path = get_model_path(parsed_args)
    parsed_args.func(env, save_path, parsed_args)


if __name__ == '__main__':
    main()
