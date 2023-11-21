from typing import Callable, cast
from unittest import TestCase

from numpy.random import Generator, default_rng

from incomplete_cooperative.bounds import compute_bounds_superadditive
from incomplete_cooperative.coalitions import (Coalition, all_coalitions,
                                               grand_coalition)
from incomplete_cooperative.exploitability import compute_exploitability
from incomplete_cooperative.generators import (
    GENERATORS, convex_generator, factory_cheerleader_next_generator,
    factory_generator, predictible_factory_generator)
from incomplete_cooperative.graph_game import GraphCooperativeGame
from incomplete_cooperative.normalize import normalize_game
from incomplete_cooperative.protocols import Game, Value


class GeneratorsTests:
    generator: Callable[[int], Callable[[int, Generator], Game]]
    is_random: bool = True
    implements_generator: bool = True

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

        for players in range(5, 10):
            game_1 = self.generator()(players)
            self.assertFalse(all(game_1 == self.generator()(players) for _ in range(100000)))

    def test_randomness_generator(self):
        if not self.implements_generator:
            cast(TestCase, self).skipTest("Not a random generator")

        for players in range(3, 10):
            game_1 = self.generator()(players, default_rng(42))
            self.assertTrue(all(game_1 == self.generator()(players, default_rng(42)) for _ in range(100)))


class TestFactoryGenerator(GeneratorsTests, TestCase):
    generator = lambda x: factory_generator  # noqa: E731

    def test_factory_pre_set_owner(self):
        players_range = range(3, 10)
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
    generator = lambda x: GENERATORS["graph"]  # noqa: E731


class TestConvexGenerator(GeneratorsTests, TestCase):
    generator = lambda x: convex_generator  # noqa: E731
    implements_generator = False

    def test_is_normalized(self):
        for players in range(3, 10):
            game = convex_generator(players)
            self.assertEqual(game.get_value(grand_coalition(game)), 1)

    def test_is_convex(self):
        for players in range(3, 10):
            with self.subTest(players=players):
                game = convex_generator(players)
                for S in all_coalitions(players):
                    for T in all_coalitions(players):
                        with self.subTest(S=S, T=T):
                            lhs = game.get_value(S) + game.get_value(T)
                            rhs = game.get_value(S & T) + game.get_value(S | T)
                            self.assertLessEqual(lhs, rhs)


class TestCheerleaderGenerator(GeneratorsTests, TestCase):
    generator = lambda x: factory_cheerleader_next_generator  # noqa: E731


class TestPredictibleFactoryGenerator(GeneratorsTests, TestCase):
    generator = lambda x: predictible_factory_generator  # noqa: E731
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
    generator = lambda x: GENERATORS["graph_random"]


class TestInternetGraphGenerator(GeneratorsTests, TestCase):
    generator = lambda x: GENERATORS["graph_internet"]


class TestGeometricGraphGenerator(GeneratorsTests, TestCase):
    generator = lambda x: GENERATORS["graph_geometric"]


class TestGeographicalGraphGenerator(GeneratorsTests, TestCase):
    generator = lambda x: GENERATORS["graph_geographical_treshold"]


class TestCycleGraphGenerator(GeneratorsTests, TestCase):
    generator = lambda x: GENERATORS["graph_cycle"]
