from pathlib import Path
from typing import Callable, cast
from unittest import TestCase

from numpy.random import Generator, default_rng

from incomplete_cooperative.bounds import compute_bounds_superadditive
from incomplete_cooperative.coalitions import (Coalition, all_coalitions,
                                               grand_coalition)
from incomplete_cooperative.exploitability import compute_exploitability
from incomplete_cooperative.game_properties import is_superadditive
from incomplete_cooperative.generators import (
    GENERATORS, FileGenerator, additive, convex_generator,
    factory_cheerleader_next_generator, factory_generator,
    predictible_factory_generator, xos)
from incomplete_cooperative.graph_game import GraphCooperativeGame
from incomplete_cooperative.normalize import normalize_game
from incomplete_cooperative.protocols import Game, Value

EPSILON = 1e-10


class GeneratorsTests:
    def generator(self):  # pragma: no cover
        raise NotImplementedError("Subclasses must implement the generator method")

    is_random: bool = True
    implements_generator: bool = True

    def test_superadditive(self):
        for players in range(3, 7):
            game = self.generator()(players)
            self.assertTrue(is_superadditive(game))

    def test_value_types(self):
        for players in range(3, 10):
            self.assertEqual(self.generator()(players).get_values().dtype, Value)

    def test_graph_game_matrix_type(self):
        if not isinstance(self.generator()(5), GraphCooperativeGame):
            self.skipTest("Not a graph game.")
        for players in range(3, 10):
            self.assertEqual(self.generator()(players)._graph_matrix.dtype, Value)

    def test_size_correct(self):
        for players in range(3, 10):
            game = self.generator()(players)
            self.assertEqual(game.number_of_players, players)

    def test_randomness(self):
        if not self.is_random:  # pragma: no cover
            cast(TestCase, self).skipTest("Not a random generator")

        for players in range(5, 8):
            game_1 = self.generator()(players)
            self.assertFalse(all(game_1 == self.generator()(players) for _ in range(1000)))

    def test_randomness_generator(self):
        if not self.implements_generator:
            cast(TestCase, self).skipTest("Not a random generator")

        for players in range(3, 7):
            game_1 = self.generator()(players, default_rng(42))
            self.assertTrue(all(game_1 == self.generator()(players, default_rng(42)) for _ in range(100)))


class TestFactoryGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return factory_generator  # noqa: E731

    def test_factory_pre_set_owner(self):
        players_range = range(3, 8)
        for players in players_range:
            for owner in range(players):
                factory = factory_generator(players, owner=owner)
                for coalition in all_coalitions(factory):
                    with self.subTest(coalition=coalition):
                        if owner not in coalition:
                            self.assertEqual(factory.get_value(coalition), 0)
                        else:
                            self.assertEqual(factory.get_value(coalition), (len(coalition) - 1))

    def test_factory_values_same_regardless_of_owner(self):
        players_range = range(3, 10)
        for players in players_range:
            for owner in range(players):
                factory_zero = factory_generator(players, owner=0)
                factory_other = factory_generator(players, owner=owner)
                for coalition in all_coalitions(players):
                    if factory_zero.get_value(coalition) and factory_other.get_value(coalition):
                        self.assertEqual(factory_zero.get_value(coalition),
                                         factory_other.get_value(coalition))

    def test_factory_values_same_regardless_of_owner_norm(self):
        players_range = range(3, 10)
        for players in players_range:
            for owner in range(players):
                factory_zero = factory_generator(players, owner=0)
                factory_other = factory_generator(players, owner=owner)
                normalize_game(factory_zero)
                normalize_game(factory_other)
                for coalition in all_coalitions(players):
                    if factory_zero.get_value(coalition) and factory_other.get_value(coalition):
                        self.assertEqual(factory_zero.get_value(coalition),
                                         factory_other.get_value(coalition))

    def test_factory_exploitability_same_with_equiv_known(self):
        players_range = range(4, 8)
        for players in players_range:
            known_coalitions = set((Coalition(2**x) for x in range(players))).union(
                [Coalition(0), grand_coalition(players)])
            factory_zero = factory_generator(players, owner=0, bounds_computer=compute_bounds_superadditive)
            factory_zero_forget = factory_generator(players, owner=0, bounds_computer=compute_bounds_superadditive)
            factory_zero_forget.set_known_values(
                factory_zero.get_known_values(known_coalitions), known_coalitions)
            factory_zero_forget.reveal_value(
                factory_zero.get_value(Coalition.from_players(set(range(1, players)))),
                Coalition.from_players(set(range(1, players))))
            for owner in range(players):
                factory_other = factory_generator(players, owner=owner, bounds_computer=compute_bounds_superadditive)
                factory_other_forget = factory_generator(players, owner=owner,
                                                         bounds_computer=compute_bounds_superadditive)
                factory_other_forget.set_known_values(
                    factory_other.get_values(known_coalitions), known_coalitions)
                reveal_coalition = Coalition.from_players(filter(
                    lambda x: x != owner,
                    range(players)))
                factory_other_forget.reveal_value(factory_other.get_value(reveal_coalition), reveal_coalition)
                factory_other_forget.compute_bounds()
                factory_zero_forget.compute_bounds()
                self.assertEqual(compute_exploitability(factory_other_forget),
                                 compute_exploitability(factory_zero_forget))

    def test_factory_exploitability_same_with_equiv_known_normalized(self):
        players_range = range(4, 8)
        for players in players_range:
            known_coalitions = set((Coalition(2**x) for x in range(players))).union(
                [Coalition(0), grand_coalition(players)])
            factory_zero = factory_generator(players, owner=0, bounds_computer=compute_bounds_superadditive)
            normalize_game(factory_zero)
            factory_zero_forget = factory_generator(players, owner=0, bounds_computer=compute_bounds_superadditive)
            normalize_game(factory_zero_forget)
            factory_zero_forget.set_known_values(
                factory_zero.get_known_values(known_coalitions), known_coalitions)
            factory_zero_forget.reveal_value(
                factory_zero.get_value(Coalition.from_players(set(range(1, players)))),
                Coalition.from_players(set(range(1, players))))
            for owner in range(players):
                factory_other = factory_generator(players, owner=owner, bounds_computer=compute_bounds_superadditive)
                normalize_game(factory_other)
                factory_other_forget = factory_generator(players, owner=owner,
                                                         bounds_computer=compute_bounds_superadditive)
                normalize_game(factory_other_forget)
                factory_other_forget.set_known_values(
                    factory_other.get_values(known_coalitions), known_coalitions)
                reveal_coalition = Coalition.from_players(filter(
                    lambda x: x != owner,
                    range(players)))
                factory_other_forget.reveal_value(factory_other.get_value(reveal_coalition), reveal_coalition)
                factory_other_forget.compute_bounds()
                factory_zero_forget.compute_bounds()
                self.assertEqual(compute_exploitability(factory_other_forget),
                                 compute_exploitability(factory_zero_forget))


class TestGraphGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return GENERATORS["graph"]  # noqa: E731
    implements_generator = False


class TestConvexGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return convex_generator  # noqa: E731
    implements_generator = False

    def test_is_normalized(self):
        for players in range(3, 10):
            game = convex_generator(players)
            self.assertEqual(game.get_value(grand_coalition(game)), 1)

    def test_is_convex(self):
        for players in range(3, 8):
            with self.subTest(players=players):
                game = convex_generator(players)
                for S in all_coalitions(players):
                    for T in all_coalitions(players):
                        with self.subTest(S=S, T=T):
                            lhs = game.get_value(S) + game.get_value(T)
                            rhs = game.get_value(S & T) + game.get_value(S | T)
                            self.assertLessEqual(lhs, rhs)


class TestCheerleaderGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return factory_cheerleader_next_generator  # noqa: E731


class TestPredictibleFactoryGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return predictible_factory_generator  # noqa: E731
    implements_generator = False

    def test_loop_around(self):
        for players in range(3, 10):
            games_1 = [predictible_factory_generator(players) for _ in range(players)]
            games_2 = [predictible_factory_generator(players) for _ in range(players)]
            for i in range(players):
                for j in range(players):
                    self.assertEqual(games_1[i], games_2[i])
                    if i != j:
                        self.assertNotEqual(games_1[i], games_1[j])


class TestRandomGraphGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return GENERATORS["graph_random"]


class TestInternetGraphGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return GENERATORS["graph_internet"]


class TestGeometricGraphGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return GENERATORS["graph_geometric"]


class TestGeographicalGraphGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return GENERATORS["graph_geographical_treshold"]


class TestCycleGraphGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return GENERATORS["graph_cycle"]


class TestAdditiveGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return additive

    def test_deterministic_singletons(self):
        singleton_values = [1, 2, 3, 4]
        expected_game_values = [
            0, 1, 2, 3, 3, 4, 5, 6, 4, 5, 6, 7, 7, 8, 9, 10
        ]
        generator = lambda x: singleton_values.pop(0)
        self.assertEqual(list(self.generator()(4, weights_dist_fn=generator).get_values()), expected_game_values)


class TestXOSGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return xos

    def test_single_is_identity(self):
        for i in range(2, 10):
            with self.subTest(number_of_players=i):
                additive_games = [additive(i) for _ in range(10)]
                first_game = additive_games[0]
                gen = lambda x, y: additive_games.pop(0)
                self.assertEqual(list(self.generator()(i, number_of_additive=1, additive_gen=gen, normalize=False)
                                      .get_values()),
                                 [-x for x in first_game.get_values()])

    def test_is_normalized(self):
        for i in range(2, 10):
            with self.subTest(number_of_players=i):
                self.assertEqual(
                    self.generator()(i, normalize=True)
                    .get_value(grand_coalition(i)),
                    -1)

    def test_normalize_additive(self):
        for i in range(2, 10):
            with self.subTest(number_of_players=i):
                additive_game = additive(i)
                gen = lambda x, y: additive_game
                self.assertEqual(
                    list(self.generator()(
                        i, number_of_additive=1, normalize_additive=True,
                        additive_gen=gen, normalize=False)
                        .get_values()),
                    list(self.generator()(
                        i, number_of_additive=1, normalize_additive=False,
                        additive_gen=gen, normalize=True)
                        .get_values()))


class TestXOSOtherGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return GENERATORS["xos12"]


class TestXSGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return GENERATORS["xs"]


class TestOXSGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return GENERATORS["oxs"]


class TestXS6Generator(GeneratorsTests, TestCase):
    def generator(self):
        return GENERATORS["xs6"]


class TestXOSOneGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return GENERATORS["xos_one"]
    is_random = False


class TestFileGenerator(GeneratorsTests, TestCase):
    def generator(self):
        return FileGenerator(Path("incomplete_cooperative/tests/test_file.npy"))
