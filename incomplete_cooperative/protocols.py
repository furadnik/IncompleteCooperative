"""Protocols specifying `Game`s."""
from __future__ import annotations

import sys
from typing import (TYPE_CHECKING, Any, Callable, Iterable, Literal, Protocol,
                    TypeAlias, runtime_checkable)

import numpy as np

if sys.version_info >= (3, 11):  # pragma: no cover
    from typing import Self
else:  # pragma: no cover
    from typing_extensions import Self

if TYPE_CHECKING:  # pragma: no cover
    from .coalitions import Coalition
    from .run.model import ModelInstance

Player: TypeAlias = int
Value: TypeAlias = np.float64
ValueIn: TypeAlias = Value | float | int
Values: TypeAlias = np.ndarray[Any, np.dtype[Value]]


@runtime_checkable
class Game(Protocol):
    """The general game protocol."""

    @property
    def number_of_players(self) -> int:
        """The number of players in the game."""

    def get_values(self, coalitions: Iterable[Coalition] | None = None) -> Values:
        """Get values for (some) coalitions the game. Defaults to all coalitions."""

    def get_value(self, coalition: Coalition) -> Value:
        """Get value for a specific coalition."""

    def copy(self) -> Self:
        """Return a deep copy of the game."""

    def __add__(self, other: Self) -> Self:
        """Add games of the same type."""


class MutableGame(Game, Protocol):
    """Game with mutable values."""

    def set_values(self, values: Values,
                   coalitions: Iterable[Coalition] | None = None) -> None:
        """Set values of (some) coalitions the game. Defaults to all coalitions."""

    def set_value(self, value: ValueIn, coalition: Coalition) -> None:
        """Set value of a specific coalition."""


@runtime_checkable
class IncompleteGame(Game, Protocol):
    """Incomplete game."""

    def get_upper_bound(self, coalition: Coalition) -> Value:
        """Get upper bound for a coalition."""

    def get_upper_bounds(self, coalitions: Iterable[Coalition] | None = None) -> Values:
        """Get upper bounds for coalitions."""

    def get_lower_bound(self, coalition: Coalition) -> Value:
        """Get lower bound for a coalition."""

    def get_lower_bounds(self, coalitions: Iterable[Coalition] | None = None) -> Values:
        """Get upper bounds for coalitions."""

    def get_interval(self, coalition: Coalition) -> np.ndarray[tuple[Literal[2]], np.dtype[Value]]:
        """Get the interval, ie. both the upper and lower bounds."""

    def get_intervals(self, coalitions: Iterable[Coalition] | None = None
                      ) -> np.ndarray[tuple[Any, Literal[2]], np.dtype[Value]]:
        """Get the intervals for (some) coalitions. Defaults to all."""

    def is_value_known(self, coalition: Coalition) -> bool:
        """Decide whether the value for a coalition is known."""

    def are_values_known(self, coalitions: Iterable[Coalition] | None = None) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """Decide whether the values for coalitions are known."""

    def get_known_value(self, coalition: Coalition) -> Value | None:
        """Get an iterable of `Value`s, or `None` if not known."""

    def get_known_values(self, coalitions: Iterable[Coalition] | None = None) -> np.ndarray[Any, np.dtype[Value]]:
        """Get an iterable of `Value`s, or `None` if not known."""

    def compute_bounds(self) -> None:
        """Compute the bounds of the ICG."""


class MutableIncompleteGame(IncompleteGame, MutableGame, Protocol):
    """Incomplete cooperative game with mutable values."""

    def set_known_values(self, known_values: Iterable[ValueIn], coalitions: Iterable[Coalition] | None = None) -> None:
        """Set known values to the selected. Drop other values."""

    def reveal_value(self, value: ValueIn, coalition: Coalition) -> None:
        """Reveal a previously unknown value of coalition."""

    def unreveal_value(self, coalition: Coalition) -> None:
        """Unreveal a previously known value of coalition."""

    def unset_value(self, coalition: Coalition) -> None:
        """Unset a previously known value of coalition."""


class BoundableIncompleteGame(IncompleteGame, Protocol):
    """An incomplete cooperative game to which we can set bounds."""

    def set_values(self, values: Values,
                   coalitions: Iterable[Coalition] | None = None) -> None:
        """Set values of (some) coalitions the game. Defaults to all coalitions."""

    def set_value(self, value: ValueIn, coalition: Coalition) -> None:
        """Set value of a specific coalition."""

    def set_upper_bounds(self, values: Values,
                         coalitions: Iterable[Coalition] | None = None) -> None:
        """Set values of (some) coalitions the game. Defaults to all coalitions."""

    def set_upper_bound(self, value: ValueIn, coalition: Coalition) -> None:
        """Set value of a specific coalition."""

    def set_lower_bounds(self, values: Values,
                         coalitions: Iterable[Coalition] | None = None) -> None:
        """Set values of (some) coalitions the game. Defaults to all coalitions."""

    def set_lower_bound(self, value: ValueIn, coalition: Coalition) -> None:
        """Set value of a specific coalition."""


GameBoundsComputer = Callable[[BoundableIncompleteGame], None]
GameGenerator = Callable[[int], Game]


State: TypeAlias = np.ndarray
Info: TypeAlias = dict[str, Any]
StepResult: TypeAlias = tuple[np.ndarray[Any, np.dtype[Value]], Value, bool, bool, Info]


class Gym(Protocol):
    """Our extended Gym protocol."""

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[State, Info]:
        """Reset the game into initial state."""

    def step(self, action: int) -> StepResult:
        """Implement one step of the arbitor, reveal coalition and compute exploitability.

        Return the new state, reward, whether we're done, and some (empty) additional info.
        """

    def unstep(self, action: int) -> StepResult:
        """Undo a step of the arbitor.

        Return the new state, reward, whether we're done, and some (empty) additional info.
        """

    def action_masks(self) -> np.ndarray:
        """Get valid actions for the agent."""

    @property
    def incomplete_game(self) -> MutableIncompleteGame:
        """The incomplete game."""

    @property
    def initially_known_coalitions(self) -> list[Coalition]:
        """A list of coalitions that are initially known."""

    @property
    def explorable_coalitions(self) -> list[Coalition]:
        """A list of coalitions that aren't initially known."""

    @property
    def reward(self) -> Value:
        """Compute the current reward and return it."""

    def get_wrapper_attr(self, _: str) -> Any:
        """Replace the __attr__ old approach."""


GymGenerator = Callable[[], Gym]


class Solver(Protocol):
    """A solver of incomplete games.

    Unlike the model, a `Solver` has complete information about the underlying full game.
    """

    def __init__(self, instance: 'ModelInstance' | None = None) -> None:
        """Initialize the solver with a model instance."""

    def next_step(self, gym: Gym) -> int:
        """Get the next move.

        This is meant to be close to the interface of `gym.Env`.
        """

    def after_reset(self, gym: Gym) -> None:
        """Get the gym state after reset.

        The subsequent calls to next_step are guaranteed to be within the same reset window.
        """


GapFunction = Callable[[IncompleteGame], Value]
