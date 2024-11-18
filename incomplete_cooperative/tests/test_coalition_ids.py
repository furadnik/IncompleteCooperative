import numpy as np

from incomplete_cooperative.coalition_ids import players, sub_coalitions


def test_players():
    assert list(players(2, 5)) == [1]
    assert list(players(3, 5)) == [0, 1]
    assert list(players(5, 5)) == [0, 2]
    assert list(players(0, 5)) == []


def test_sub_coalition():
    assert list(sub_coalitions(2, 5)) == [0, 2]
    assert list(sub_coalitions(3, 5)) == [0, 1, 2, 3]
    assert list(sub_coalitions(5, 5)) == [0, 1, 4, 5]
