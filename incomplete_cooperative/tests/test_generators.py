from unittest import TestCase

from incomplete_cooperative.coalitions import all_coalitions
from incomplete_cooperative.generators import factory_generator
from incomplete_cooperative.normalize import normalize_game
from incomplete_cooperative.coalitions import Coalition, grand_coalition
from incomplete_cooperative.bounds import compute_bounds_superadditive
from incomplete_cooperative.exploitability import compute_exploitability


class TestFactoryGenerator(TestCase):

    def test_factory_pre_set_owner(self):
        players_range = range(3, 10)
        for players in players_range:
            for owner in range(players):
                factory = factory_generator(players, owner)
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
                factory_zero = factory_generator(players, 0)
                factory_other = factory_generator(players, owner)
                for coalition in all_coalitions(players):
                    if factory_zero.get_value(coalition) and factory_other.get_value(coalition):
                        self.assertEqual(factory_zero.get_value(coalition),
                                         factory_other.get_value(coalition))

    def test_factory_values_same_regardless_of_owner_norm(self):
        players_range = range(3, 10)
        for players in players_range:
            for owner in range(players):
                factory_zero = factory_generator(players, 0)
                factory_other = factory_generator(players, owner)
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
            factory_zero = factory_generator(players, 0, compute_bounds_superadditive)
            factory_zero_forget = factory_generator(players, 0, compute_bounds_superadditive)
            factory_zero_forget.set_known_values(
                factory_zero.get_known_values(known_coalitions), known_coalitions)
            factory_zero_forget.reveal_value(
                factory_zero.get_value(Coalition.from_players(set(range(1, players)))),
                Coalition.from_players(set(range(1, players))))
            for owner in range(players):
                factory_other = factory_generator(players, owner, compute_bounds_superadditive)
                factory_other_forget = factory_generator(players, owner, compute_bounds_superadditive)
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
            factory_zero = factory_generator(players, 0, compute_bounds_superadditive)
            normalize_game(factory_zero)
            factory_zero_forget = factory_generator(players, 0, compute_bounds_superadditive)
            normalize_game(factory_zero_forget)
            factory_zero_forget.set_known_values(
                factory_zero.get_known_values(known_coalitions), known_coalitions)
            factory_zero_forget.reveal_value(
                factory_zero.get_value(Coalition.from_players(set(range(1, players)))),
                Coalition.from_players(set(range(1, players))))
            for owner in range(players):
                factory_other = factory_generator(players, owner, compute_bounds_superadditive)
                normalize_game(factory_other)
                factory_other_forget = factory_generator(players, owner, compute_bounds_superadditive)
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
