"""This is a script for comparing the algorithms for multiplicative estimation with our approach."""
from os import getenv

from incomplete_cooperative.bounds import BOUNDS
from incomplete_cooperative.coalitions import (Coalition,
                                               minimal_game_coalitions)
from incomplete_cooperative.game import IncompleteCooperativeGame
from incomplete_cooperative.generators import GENERATORS

OUR_ACTION_SEQUENCE: list[Coalition] = [Coalition(int(x)) for x in (getenv("ACTION_SEQUENCE") or "0").split(",")]
GAME_GENERATOR = GENERATORS.get(getenv("GAME_CLASS") or "k_budget")
NUMBER_OF_PLAYERS = getenv("NUMBER_OF_PLAYERS") or 6
BOUNDS_COMPUTER = BOUNDS.get(getenv("BOUNDS_COMPUTER") or "sam_apx_1")


def main() -> None:
    """Compare multiplicative algos to ours."""
    game = GAME_GENERATOR(NUMBER_OF_PLAYERS)
    incomplete_game = IncompleteCooperativeGame(NUMBER_OF_PLAYERS, BOUNDS_COMPUTER)
    minimal_game = minimal_game_coalitions(NUMBER_OF_PLAYERS)
