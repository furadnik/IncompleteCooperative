"""CLI."""
import multiprocessing
import random
import sys
from argparse import ArgumentParser

from .run.eval import add_eval_parser
from .run.learn import add_learn_parser
from .run.model import ModelInstance, add_model_arguments

COMMANDS = {
    "learn": add_learn_parser,
    "eval": add_eval_parser
}


def get_argument_parser(commands=COMMANDS) -> ArgumentParser:
    """Get argument parser."""
    ap = ArgumentParser(description="Allow usage of the ICG module from the command-line.")
    # ap.add_argument("--seed", required=False,
    #                 help="Set the seed for possible random generation.")
    # ap.add_argument("-hs", action="store_true")
    add_model_arguments(ap)

    if commands:
        subparsers = ap.add_subparsers(required=True)
        for command, add_parser in commands.items():
            subparser = subparsers.add_parser(command)
            add_parser(subparser)

    return ap


def main(ap: ArgumentParser = get_argument_parser(),
         args: list[str] = sys.argv) -> None:
    """Run the main script."""
    args = args[1:]
    parsed_args = ap.parse_args(args)

    if parsed_args.hs:
        print(";)")
        sys.exit(69)

    # if parsed_args.seed is not None:  # TODO: does not apply to the agent yet
    #     random.seed(parsed_args.seed)

    print(parsed_args)
    instance = ModelInstance.from_parsed_arguments(parsed_args)
    parsed_args.func(instance, parsed_args)


if __name__ == '__main__':  # pragma: no cover
    multiprocessing.set_start_method("forkserver")
    main()
