"""Learning script."""
from .model import ModelInstance


def learn_func(instance: ModelInstance, args) -> None:
    """Model learning."""
    model = instance.model
    model.learn(args.learn_total_timesteps)
    instance.save(model)


def add_learn_parser(parser) -> None:
    """Add `learn` command to the CLI argument parser."""
    parser.add_argument("--learn-total-timesteps", default=50_000, type=int, help="Total timesteps taken by the agent.")
    parser.set_defaults(func=learn_func)
