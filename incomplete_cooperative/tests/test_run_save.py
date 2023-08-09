"""Testing module for `save`."""
from unittest import TestCase
from argparse import Namespace
from pathlib import Path
import datetime
import numpy as np
import json
import tempfile
from incomplete_cooperative.run.save import json_serializer, save_json, Output


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


class TestJsonSaver(TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "testing"

    def getOutput(self, exploitability=np.fromiter([1, 2], float), actions=np.fromiter([0, 1], int),
                  parsed_args=None) -> Output:
        """Get sample output."""
        if parsed_args is None:
            parsed_args = Namespace(foo="bar", baz=42, func="eval")
        return Output(exploitability, actions, parsed_args)

    def assertHasJson(self, path: Path, expected: dict) -> None:
        with path.open("r") as f:
            self.assertDictEqual(json.load(f), expected)

    def test_save_first(self):
        save_json(self.path, "foobar", self.getOutput())
        self.assertHasJson(self.path, {"foobar": {"metadata": {"foo": "bar", "baz": 42, "run_type": "eval"},
                                                  "exploitability": [1.0, 2.0],
                                                  "actions": [0, 1]}})

    def test_save_second(self):
        self.maxDiff = None
        save_json(self.path, "foobar", self.getOutput())
        save_json(self.path, "baz", self.getOutput(actions=np.fromiter([3, 4], int)))
        self.assertHasJson(self.path, {"foobar": {"metadata": {"foo": "bar", "baz": 42, "run_type": "eval"},
                                                  "exploitability": [1.0, 2.0],
                                                  "actions": [0, 1]},
                                       "baz": {"metadata": {"foo": "bar", "baz": 42, "run_type": "eval"},
                                               "exploitability": [1.0, 2.0],
                                               "actions": [3, 4]}})

    def test_save_second_same_name(self):
        save_json(self.path, "foobar", self.getOutput())
        save_json(self.path, "foobar", self.getOutput(actions=np.fromiter([3, 4], int)))
        self.assertHasJson(self.path, {"foobar": {"metadata": {"foo": "bar", "baz": 42, "run_type": "eval"},
                                                  "exploitability": [1.0, 2.0],
                                                  "actions": [0, 1]}})
