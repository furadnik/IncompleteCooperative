"""Testing module for `save`."""
from argparse import Namespace
from incomplete_cooperative.run.save import json_serializer, save_json, Output, save_exploitability_plot, SAVERS
from pathlib import Path
from typing import cast
from unittest import TestCase
import datetime
import json
import numpy as np
import tempfile


class TestJsonSerializer(TestCase):
    stuff = [
        (Path("/x/y/z"), "/x/y/z"),
        (datetime.date(year=2020, month=12, day=10), "datetime.date(2020, 12, 10)"),
    ]

    def test_stuff(self):
        for obj, expected in self.stuff:
            with self.subTest(obj=obj):
                self.assertEqual(json.loads(json.dumps({"data": obj}, default=json_serializer)),
                                 {"data": expected})


class TestSaverMixin:

    func: str

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        cast(TestCase, self).addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "testing"

    def getOutput(self, exploitability=np.array([[3, 1, 2]]), actions=np.array([[0, 1]]),
                  parsed_args=None) -> Output:
        """Get sample output."""
        if parsed_args is None:
            parsed_args = Namespace(foo="bar", baz=42, func="eval")
        return Output(exploitability, actions, parsed_args)

    def assertHasJson(self, path: Path, expected: dict) -> None:
        with path.open("r") as f:
            cast(TestCase, self).assertDictEqual(json.load(f), expected)

    def test_call_function(self):
        self.path.mkdir()
        self.assertEqual(len(list(self.path.iterdir())), 0)
        cast(TestCase, self).assertIn(self.func, SAVERS.keys())
        SAVERS[self.func](self.path / self.func, "asdfg", self.getOutput())
        self.assertEqual(list(x.name for x in self.path.iterdir()), [self.func])


class TestJsonSaver(TestSaverMixin, TestCase):

    func = "data.json"

    def test_save_first(self):
        save_json(self.path, "foobar", self.getOutput())
        self.assertHasJson(self.path, {"foobar": {"metadata": {"foo": "bar", "baz": 42, "run_type": "eval"},
                                                  "exploitability": [[3, 1, 2]],
                                                  "actions": [[0, 1]]}})

    def test_save_second(self):
        self.maxDiff = None
        save_json(self.path, "foobar", self.getOutput())
        save_json(self.path, "baz", self.getOutput(actions=np.array([[3, 4]])))
        self.assertHasJson(self.path, {"foobar": {"metadata": {"foo": "bar", "baz": 42, "run_type": "eval"},
                                                  "exploitability": [[3, 1, 2]],
                                                  "actions": [[0, 1]]},
                                       "baz": {"metadata": {"foo": "bar", "baz": 42, "run_type": "eval"},
                                               "exploitability": [[3, 1, 2]],
                                               "actions": [[3, 4]]}})

    def test_save_second_same_name(self):
        save_json(self.path, "foobar", self.getOutput())
        save_json(self.path, "foobar", self.getOutput(actions=np.array([[3, 4]])))
        self.assertHasJson(self.path, {"foobar": {"metadata": {"foo": "bar", "baz": 42, "run_type": "eval"},
                                                  "exploitability": [[3, 1, 2]],
                                                  "actions": [[0, 1]]}})


class TestExplPlotSave(TestSaverMixin, TestCase):

    func = "exploitability_plots"

    def test_save_plots(self):
        save_exploitability_plot(self.path, "asdf", self.getOutput())
        self.assertTrue(self.path.exists())
        self.assertTrue((self.path / "asdf.png").exists())
        self.assertTrue((self.path / "asdf.png").is_file())

    def test_save_plots_multiple(self):
        save_exploitability_plot(self.path, "asdfg", self.getOutput())
        save_exploitability_plot(self.path, "asdf", self.getOutput())
        self.assertTrue(self.path.exists())
        self.assertTrue((self.path / "asdf.png").exists())
        self.assertTrue((self.path / "asdf.png").is_file())
