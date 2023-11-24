"""Check if a game is supermodular."""
from .coalitions import Coalition, all_coalitions, grand_coalition
from .protocols import Game, IncompleteGame, Player


def check_supermodularity(game: IncompleteGame | Game) -> tuple[Coalition, Coalition, Player] | None:
    """Return the violating coalitions and player; or None if the game is supermodular."""
    for T in all_coalitions(game):
        for S in filter(lambda s: s | T == T, all_coalitions(game)):
            for i in filter(lambda i: i not in T, grand_coalition(game).players):
                if isinstance(game, IncompleteGame):
                    if any([
                        not game.is_value_known(S | Coalition.from_players({i})),
                        not game.is_value_known(T | Coalition.from_players({i})),
                        not game.is_value_known(S),
                        not game.is_value_known(T),
                    ]):
                        continue

                lhs = game.get_value(S | Coalition.from_players({i})) - game.get_value(S)
                rhs = game.get_value(T | Coalition.from_players({i})) - game.get_value(T)
                if lhs > rhs:
                    return T, S, i
    return None
