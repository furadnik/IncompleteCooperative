from unittest import TestCase

from incomplete_cooperative.coalitions import all_coalitions
from incomplete_cooperative.generators import factory_generator


class TestFactoryGenerator(TestCase):

    def test_factory_pre_set_owner(self):
        factory = factory_generator(10, 3)
        for coalition in all_coalitions(factory):
            with self.subTest(coalition=coalition):
                if 3 not in coalition:
                    self.assertEqual(factory.get_value(coalition), 0)
                else:
                    self.assertEqual(factory.get_value(coalition), (len(coalition) - 1)**2)
