"""Tests for the `run` module."""
from argparse import ArgumentParser
from os import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from incomplete_cooperative.random_player import RandomPolicy
from incomplete_cooperative.run.eval import add_eval_parser, eval_func
from incomplete_cooperative.run.learn import add_learn_parser, learn_func
from incomplete_cooperative.run.model import ModelInstance, add_model_arguments


class TestAddModelArguments(TestCase):

    @property
    def ap(self):
        """Argument Parser with valid args."""
        ap = ArgumentParser()
        add_model_arguments(ap)
        return ap

    def test_valid_ap(self):
        self.ap.parse_args(["--model-name", "asdf"])

    def test_valid_arguments(self):
        tests = [
            ([], lambda x: x.name == "icg"),
            (["--model-name", "asdf"], lambda x: x.name == "asdf"),
            (["--number-of-players", "42"], lambda x: x.number_of_players == 42),
            (["--steps-per-update", "420"], lambda x: x.steps_per_update == 420),
            (["--game-generator", "factory"], lambda x: x.game_generator == "factory"),
        ]
        for arguments, test in tests:
            with self.subTest(args=arguments):
                parsed_args = self.ap.parse_args(arguments)
                model = ModelInstance.from_parsed_arguments(parsed_args)
                self.assertTrue(test(model))


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

    def test_get_random_model(self):
        model = ModelInstance(random=True).model
        self.assertIsInstance(model.policy, RandomPolicy)


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

    def test_function_proper(self):
        self.assertEqual(self.parsed_args.func, eval_func)

    def test_run_eval(self):
        eval_func(self.model, self.parsed_args)  # TODO: implement later.
        self.assertEqual(len(list(Path(self._tmp.name).iterdir())), 2)

    def test_both_parsed(self):
        parsed = self.ap.parse_args(["--eval-repetitions", "1", "--eval-episode-length", "2"])
        self.assertEqual(parsed.eval_repetitions, 1)
        self.assertEqual(parsed.eval_episode_length, 2)
