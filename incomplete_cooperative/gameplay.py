"""A module containing helpers for manipulating the `Game`s."""
import time
from itertools import chain, combinations
from multiprocessing import Pool
from typing import Any, Iterable, Sequence

import numpy as np

from .coalitions import Coalition, all_coalitions, get_known_coalitions
from .exploitability import compute_exploitability
from .protocols import (Game, GameGenerator, IncompleteGame,
                        MutableIncompleteGame, Value)

Action = Coalition  # we are choosing a coalition
ActionSequence = list[Action]


def possible_next_actions(game: IncompleteGame) -> Iterable[Action]:
    """Get an interable of the next possible actions as defined in a `Gymnasium` context."""
    return (x for x in all_coalitions(game) if not game.is_value_known(x))


def possible_action_sequences(game: IncompleteGame, max_size: int | None = None) -> Iterable[ActionSequence]:
    """Get the possible sequences of actions to take with a game.

    Arguments:
        game: The game to take the actions on.
        max_size: The maximum number of steps to take, `None` if not bounded.
    """
    possible_actions = list(possible_next_actions(game))
    max_size = max_size if max_size is not None else len(possible_actions)

    return chain.from_iterable(map(list, combinations(possible_actions, i))
                               for i in range(max_size + 1))


def apply_action_sequence(game: MutableIncompleteGame, full_game: Game,
                          action_sequence: ActionSequence, include: ActionSequence = []) -> None:
    """Apply an action sequence to a game, according to the `full_game`.

    Note: `game` is an output argument, it gets reset and modified.
    """
    if include:
        action_sequence = list(set(action_sequence).union(include))
    game.set_known_values(full_game.get_values(action_sequence), action_sequence)


def _get_act_sequence_exploitability(game: MutableIncompleteGame, full_game: Game, action_sequence: ActionSequence,
                                     known_coalitions: list[Coalition]
                                     ) -> tuple[ActionSequence, Value]:  # pragma: no cover - multiprocessing
    """Get the exploitability of an action sequence."""
    apply_action_sequence(game, full_game, action_sequence, include=known_coalitions)
    game.compute_bounds()
    return action_sequence, compute_exploitability(game)


def get_exploitabilities_of_action_sequences(
        game: MutableIncompleteGame, full_game: Game, max_size: int | None = None
) -> Iterable[tuple[ActionSequence, Value]]:
    """Get exploitabilities of action sequences."""
    known_coalitions = list(get_known_coalitions(game))
    with Pool() as p:
        return p.starmap(
            _get_act_sequence_exploitability,
            ((game, full_game, seq, known_coalitions) for seq in possible_action_sequences(game, max_size=max_size)))


def sample_exploitabilities_of_action_sequences(
        game: MutableIncompleteGame, full_game_generator: GameGenerator, samples: int = 1,
        **kwargs: Any
) -> tuple[list[ActionSequence], np.ndarray[Any, np.dtype[Value]]]:
    """Sample exploitabilities of action sequences."""
    print(int(time.time()), f"Sampling exploitabilities for {game}")

    initially_known_coalitions = list(get_known_coalitions(game))

    # get action list and initial row of the values
    full_game = full_game_generator(game.number_of_players)
    game.set_known_values(full_game.get_values(initially_known_coalitions),
                          initially_known_coalitions)
    initial_run = list(get_exploitabilities_of_action_sequences(
        game, full_game, **kwargs))
    actions = [x[0] for x in initial_run]
    initial_values = np.fromiter((x[1] for x in initial_run), Value)
    # generate an empty array of values
    values = np.zeros((samples, initial_values.shape[0]), dtype=Value)
    # add the initial values as first row
    values[0] = initial_values

    for i in range(1, samples):
        start = int(time.time())
        print(start, f"Sampling exploitabilities for {game}, sample {i+1}")
        full_game = full_game_generator(game.number_of_players)
        game.set_known_values(full_game.get_values(initially_known_coalitions),
                              initially_known_coalitions)
        values[i] = np.fromiter((x[1] for x in get_exploitabilities_of_action_sequences(game, full_game, **kwargs)),
                                Value, initial_values.shape[0])
        print(f"Took {int(time.time()) - start} seconds ({(int(time.time()) - start)/60} minutes).")
    return actions, values


def get_exploitabilities_of_action_sequence(
        game: MutableIncompleteGame, full_games: Iterable[Game],
        action_sequence: ActionSequence
) -> Iterable[Value]:
    """Apply an action sequence to an iterable of games."""
    known_coalitions = list(get_known_coalitions(game))
    with Pool() as p:
        return map(lambda x: x[1], p.starmap(
            _get_act_sequence_exploitability,
            ((game, full_game, action_sequence, known_coalitions) for full_game in full_games)))


def get_stacked_exploitabilities_of_action_sequences(
        game: MutableIncompleteGame, full_games: Sequence[Game],
        action_sequences: Iterable[ActionSequence]
) -> Iterable[np.ndarray]:
    """Get the expected exploitabilities of a list of action sequences."""
    number_of_games = len(full_games)
    for action_sequence in action_sequences:
        yield np.fromiter(
            get_exploitabilities_of_action_sequence(game, full_games, action_sequence),
            Value, count=number_of_games)
