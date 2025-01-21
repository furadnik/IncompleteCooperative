"""Tests suite of learning."""
from argparse import Namespace
from os import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import TestCase
from unittest.mock import patch

from incomplete_cooperative.run.eval import eval_func
from incomplete_cooperative.run.learn import learn_func
from incomplete_cooperative.run.model import ModelInstance
from incomplete_cooperative.run.save import Output
from incomplete_cooperative.run.solve import solve_func


class GetLearningResultMixin:
    """Mixin for getting learning results."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        cast(TestCase, self).addCleanup(self._tmp.cleanup)
        chdir(self._tmp.name)

    def get_instance(self, **kwargs) -> tuple[Namespace, ModelInstance]:
        ns = Namespace(**kwargs)
        instance = ModelInstance.from_parsed_arguments(ns)
        return ns, instance

    def _saver(self, path: Path, unique_name: str, output: Output) -> None:
        self._output = output

    def get_saver_output(self, func, instance, args) -> Output:
        """Get output of a saver from running func."""
        with patch("incomplete_cooperative.run.save.SAVERS", {"saver": self._saver}):
            func(instance, args)
        return self._output

    def get_random_results(self, **kwargs) -> Output:
        """Get results of random solver."""
        args, instance = self.get_instance(solver="random", **kwargs)
        return self.get_saver_output(solve_func, instance, args)

    def get_learning_results(self, **kwargs) -> Output:
        """Get results of learning."""
        args, instance = self.get_instance(**kwargs)
        learn_func(instance, args)
        return self.get_saver_output(eval_func, instance, args)


class LearningTester(GetLearningResultMixin):

    kwargs: dict

    def test_better_than_random(self):
        learned_output = self.get_learning_results(**self.kwargs)
        random_output = self.get_random_results(**self.kwargs)
        self.assertLess(learned_output.data_avg_final,
                        random_output.data_avg_final, msg=learned_output)

    def test_random_is_random(self):
        random_results = self.get_random_results(**self.kwargs)
        actions_list = random_results.actions.flatten().tolist()
        distinct_actions = list(set(actions_list))
        self.assertGreater(len(distinct_actions), 1, msg=actions_list)


class TestSimpleGame(LearningTester, TestCase):

    kwargs = {
        "number_of_players": 4,
        "game_generator": "factory_fixed",
        "learn_total_timesteps": 3000,
        "steps_per_update": 64,
        "eval_nondeterministic": False,
        "eval_repetitions": 10,
        "solve_repetitions": 10,
        "run_steps_limit": 1,
    }

    def test_found_optimal(self):
        args, instance = self.get_instance(**self.kwargs)
        results = self.get_learning_results(**self.kwargs)
        env = instance.get_env()

        best_reward = -100000
        best_actions = []
        for i in range(env.action_masks().shape[0]):
            _, reward, _, _, out_dict = env.step(i)
            action = out_dict["chosen_coalition"]
            env.reset()
            if reward == best_reward:
                best_actions.append(action)
            if reward > best_reward:
                best_reward = reward
                best_actions = [action]
        for action in results.actions.flatten().tolist():
            self.assertIn(action, best_actions, results.actions)


class TestSimpleGameParallel(TestSimpleGame):

    kwargs = {
        **TestSimpleGame.kwargs,
        "environment": "parallel"
    }


class TestSimpleTanh(TestSimpleGame):

    kwargs = {
        **TestSimpleGame.kwargs,
        "policy_activation_fn": "tanh",
    }


class TestSimpleParallelTanh(TestSimpleGameParallel):

    kwargs = {
        **TestSimpleGameParallel.kwargs,
        "policy_activation_fn": "tanh",
    }
