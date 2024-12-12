"""Evaluating script."""
from functools import partial

from sb3_contrib.common.maskable.utils import get_action_masks  # type: ignore

from incomplete_cooperative.evaluation import evaluate

from .model import ModelInstance
from .save import Output, save


def _eval_next_step(env, model, deterministic):
    action_masks = get_action_masks(env)
    obs = env.get_wrapper_attr("state")
    action, _ = model.predict(
        obs, action_masks=action_masks, deterministic=deterministic)
    return action


def eval_func(instance: ModelInstance, parsed_args) -> None:
    """Evaluate the model."""
    model = instance.model

    if instance.run_steps_limit is None:
        instance.run_steps_limit = 2**instance.number_of_players

    exploitability, actions_all = evaluate(
        partial(_eval_next_step, model=model, deterministic=parsed_args.eval_deterministic),
        instance.get_env, parsed_args.eval_repetitions,
        instance.run_steps_limit,
        instance.gap_function_callable,
        processes=1  # the PPO model doesn't support multiprocessing
    )

    save(instance.model_dir, instance.unique_name,
         Output(exploitability, actions_all, parsed_args))


def add_eval_parser(parser) -> None:
    """Fill in the parser with arguments for evaluating the model."""
    parser.add_argument("--eval-output-path", default=".", type=str)
    parser.add_argument("--eval-repetitions", default=100, type=int)
    parser.add_argument("--eval-deterministic", action="store_true")
    parser.set_defaults(func=eval_func)
