"""Import games from various formats."""
import csv
from io import TextIOWrapper

from .coalitions import Coalition
from .game import IncompleteCooperativeGame
from .protocols import Game


def game_from_csv(inp: TextIOWrapper) -> Game:
    """TODO: implement later."""
    number_of_players = 5
    game = IncompleteCooperativeGame(number_of_players)
    file = csv.reader(inp, delimiter=";")
    for line in file:
        if not line[1]:
            continue
        players_str, value_str = line[1], line[2]
        players_coalition = Coalition.from_players(i for i in range(number_of_players) if str(i + 1) in players_str)
        game.set_value(int(value_str), players_coalition)
        print(players_str, players_coalition, value_str)
    return game
