from unittest import TestCase
from unittest.mock import Mock, patch

from incomplete_cooperative.__main__ import get_argument_parser, main


class TestArgParser(TestCase):

    def setUp(self):
        self.mock_fn = Mock()
        self.commands = {"test": lambda x: x.set_defaults(func=self.mock_fn)}
        self.parser = get_argument_parser(commands=self.commands)

    def test_run_command(self):
        args = self.parser.parse_args(["test"])
        self.mock_fn.assert_not_called()
        args.func()
        self.mock_fn.assert_called_once()


class TestMain(TestCase):

    def test_hs(self):
        self.assertRaises(SystemExit, main, ap=Mock(hs=True), args=["-hs", "learn"])

    def test_proper_run(self):
        m = Mock()
        parsed = m.parse_args.return_value
        parsed.hs = False
        parsed.seed = None
        parsed.model_dir = "."

        main(m, [])
        m.parse_args.assert_called_once_with([])
        parsed.func.assert_called_once()

    def test_seed(self):
        m = Mock()
        parsed = m.parse_args.return_value
        parsed.hs = False
        parsed.seed = 42
        parsed.model_dir = "."

        with patch("random.seed") as p:
            main(m, [])
            p.assert_called_once_with(42)
