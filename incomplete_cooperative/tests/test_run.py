"""Tests for the `run` module."""
import json
from argparse import ArgumentParser
from os import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
from stable_baselines3.common.vec_env import SubprocVecEnv

from incomplete_cooperative.run.best_states import (add_best_states_parser,
                                                    best_states_func)
from incomplete_cooperative.run.eval import add_eval_parser, eval_func
from incomplete_cooperative.run.greedy import add_greedy_parser, greedy_func
from incomplete_cooperative.run.learn import add_learn_parser, learn_func
from incomplete_cooperative.run.model import ModelInstance, add_model_arguments
from incomplete_cooperative.run.save import SAVERS
from incomplete_cooperative.run.solve import add_solve_parser, solve_func


class TestAddModelArguments(TestCase):

    @property
    def ap(self):
        """Argument Parser with valid args."""
        ap = ArgumentParser()
        add_model_arguments(ap)
        return ap

    def test_valid_ap(self):
        self.ap.parse_args(["--number-of-players", "42"])

    def test_valid_arguments(self):
        tests = [
            ([], lambda x: x.model_dir == Path(".")),
            (["--number-of-players", "42"], lambda x: x.number_of_players == 42),
            (["--steps-per-update", "420"], lambda x: x.steps_per_update == 420),
            (["--game-generator", "factory"], lambda x: x.game_generator == "factory"),
            (["--game-generator", "factory"], lambda x: x.run_steps_limit is None),
            (["--run-steps-limit", "42"], lambda x: x.run_steps_limit == 42),
            (["--model-dir", "/foo/bar"], lambda x: x.model_dir == Path("/foo/bar")),
            (["--model-dir", "/foo/bar"], lambda x: x.model_path == Path("/foo/bar/model")),
            (["--model-path", "/foo/bar"], lambda x: x.model_path == Path("/foo/bar")),
            (["--environment", "parallel"], lambda x: x.environment_class == SubprocVecEnv),
            ([], lambda x: x.environment_class == DummyVecEnv),
        ]
        for arguments, test in tests:
            with self.subTest(args=arguments):
                parsed_args = self.ap.parse_args(arguments)
                model = ModelInstance.from_parsed_arguments(parsed_args)
                self.assertTrue(test(model), msg=model)


class TestModelInstance(TestCase):

    def setUp(self):
        self.model = ModelInstance()
        self._tmp = TemporaryDirectory()
        chdir(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_env_generator_proper_parallel_envs(self):
        self.assertEqual(self.model.env_generator().num_envs,
                         self.model.parallel_environments)

    def test_save_and_load(self):
        orig_model = self.model.model
        self.model.save(orig_model)
        self.assertTrue(self.model.model_path.with_suffix(".zip").exists())

        with patch("incomplete_cooperative.run.model.MaskablePPO") as m:
            self.model.model
            m.load.assert_called_once()


class TestLearn(TestCase):

    def setUp(self):
        self.model = ModelInstance()
        self.ap = ArgumentParser()
        add_learn_parser(self.ap)
        self.parsed_args = self.ap.parse_args(["--learn-total-timesteps", "0"])

        self._tmp = TemporaryDirectory()
        chdir(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_function_proper(self):
        self.assertEqual(self.parsed_args.func, learn_func)

    def test_saving(self):
        learn_func(self.model, self.parsed_args)
        self.assertTrue(self.model.model_path.with_suffix(".zip").exists())
        # self.assertEqual(len(list(Path(self._tmp.name).iterdir())), 3)


class TestEval(TestCase):

    def setUp(self):
        self.model = ModelInstance()
        self.ap = ArgumentParser()
        add_eval_parser(self.ap)
        self.parsed_args = self.ap.parse_args(["--eval-repetitions", "1"])

        self._tmp = TemporaryDirectory()
        chdir(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_eval_test(self, path):
        self.model.model_dir = path
        self.model.unique_name = "asdf"
        eval_func(self.model, self.parsed_args)  # TODO: implement later.
        self.assertEqual(len(list(path.iterdir())), 3)
        found = False
        self.assertEqual(set(SAVERS.keys()), set(x.name for x in path.iterdir()))
        for file in path.iterdir():
            if file.suffix == ".json":
                with file.open("r") as f:
                    self.assertEqual(list(json.load(f)["asdf"].keys()), ["data", "actions", "metadata"])
                    found = True
        self.assertTrue(found)

    def test_function_proper(self):
        self.assertEqual(self.parsed_args.func, eval_func)

    def test_run_eval(self):
        path = Path(self._tmp.name)
        self.run_eval_test(path)

    def test_run_eval_create_folder(self):
        path = Path(self._tmp.name) / "model"
        self.run_eval_test(path)

    def test_both_parsed(self):
        parsed = self.ap.parse_args(["--eval-repetitions", "1", "--eval-deterministic"])
        self.assertEqual(parsed.eval_repetitions, 1)
        self.assertTrue(parsed.eval_deterministic)

    def test_eval_deterministic(self):
        parsed = self.ap.parse_args(["--eval-repetitions", "1"])
        self.assertFalse(parsed.eval_deterministic)
        parsed = self.ap.parse_args(["--eval-deterministic"])
        self.assertTrue(parsed.eval_deterministic)


class TestSolve(TestCase):

    def setUp(self):
        self.model = ModelInstance(number_of_players=3)

        self._tmp = TemporaryDirectory()
        chdir(self._tmp.name)

    def get_parsed_args(self, name="greedy"):
        ap = ArgumentParser()
        add_solve_parser(ap)
        parsed_args = ap.parse_args(["--solve-repetitions", "1", "--solver", name])
        return parsed_args

    def tearDown(self):
        self._tmp.cleanup()

    def run_solve_test(self, path):
        self.model.model_dir = path
        self.model.unique_name = "asdf"
        solve_func(self.model, self.get_parsed_args())  # TODO: implement later.
        self.assertEqual(len(list(path.iterdir())), 3)
        found = False
        self.assertEqual(set(SAVERS.keys()), set(x.name for x in path.iterdir()))
        for file in path.iterdir():
            if file.suffix == ".json":
                with file.open("r") as f:
                    self.assertEqual(list(json.load(f)["asdf"].keys()), ["data", "actions", "metadata"])
                    found = True
        self.assertTrue(found)

    def test_function_proper(self):
        self.assertEqual(self.get_parsed_args().func, solve_func)

    def test_run_solve(self):
        path = Path(self._tmp.name)
        self.run_solve_test(path)

    def test_run_solve_create_folder(self):
        path = Path(self._tmp.name) / "model"
        self.run_solve_test(path)


class TestBestStates(TestCase):

    def setUp(self):
        self.model = ModelInstance(number_of_players=4, run_steps_limit=4)

        self._tmp = TemporaryDirectory()
        chdir(self._tmp.name)

    def get_parsed_args(self):
        ap = ArgumentParser()
        add_best_states_parser(ap)
        parsed_args = ap.parse_args([])
        return parsed_args

    def tearDown(self):
        self._tmp.cleanup()

    def run_best_states_test(self, path):
        self.model.model_dir = path
        self.model.unique_name = "asdf"
        best_states_func(self.model, self.get_parsed_args())  # TODO: implement later.
        self.assertEqual(len(list(path.iterdir())), 3)
        found = False
        self.assertEqual(set(SAVERS.keys()), set(x.name for x in path.iterdir()))
        for file in path.iterdir():
            if file.suffix == ".json":
                with file.open("r") as f:
                    self.assertEqual(list(json.load(f)["asdf"].keys()), ["data", "actions", "metadata"])
                    found = True
        self.assertTrue(found)

    def test_function_proper(self):
        self.assertEqual(self.get_parsed_args().func, best_states_func)

    def test_run_solve(self):
        path = Path(self._tmp.name)
        self.run_best_states_test(path)

    def test_run_solve_create_folder(self):
        path = Path(self._tmp.name) / "model"
        self.run_best_states_test(path)


class TestGreedy(TestCase):

    def setUp(self):
        self.model = ModelInstance(number_of_players=4, run_steps_limit=4)

        self._tmp = TemporaryDirectory()
        chdir(self._tmp.name)

    def get_parsed_args(self):
        ap = ArgumentParser()
        add_greedy_parser(ap)
        parsed_args = ap.parse_args([])
        return parsed_args

    def tearDown(self):
        self._tmp.cleanup()

    def run_greedy_test(self, path, randomize):
        self.model.model_dir = path
        self.model.unique_name = "asdf"
        greedy_func(self.model, self.get_parsed_args(), randomize)  # TODO: implement later.
        self.assertEqual(len(list(path.iterdir())), 3)
        found = False
        self.assertEqual(set(SAVERS.keys()), set(x.name for x in path.iterdir()))
        for file in path.iterdir():
            if file.suffix == ".json":
                with file.open("r") as f:
                    self.assertEqual(list(json.load(f)["asdf"].keys()), ["data", "actions", "metadata"])
                    found = True
        self.assertTrue(found)

    def test_run_solve(self):
        path = Path(self._tmp.name)
        self.run_greedy_test(path, False)

    def test_run_solve_create_folder(self):
        path = Path(self._tmp.name) / "model"
        self.run_greedy_test(path, False)

    def test_run_solve_randomize(self):
        path = Path(self._tmp.name)
        self.run_greedy_test(path, True)
