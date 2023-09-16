"""An incomplete cooperative game representation."""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Iterable

import numpy as np

from .coalitions import Coalition, all_coalitions
from .protocols import Game, Value

Coalitions = Iterable[Coalition]
CoalitionPlayers = Iterable[int]

LOGGER = logging.getLogger(__name__)


def _polish_graph_matrix(matrix: np.ndarray) -> None:
    """Polish the graph matrix."""
    for i in range(matrix.shape[0]):
        for j in range(i, matrix.shape[1]):
            matrix[j, i] = 0


class GraphCooperativeGame:
    """Represent a full graph game."""

    def __init__(self, graph_matrix: np.ndarray[Any, np.dtype[Value]]) -> None:
        """Save basic game info.

        Arguments:
            graph_matrix: The weight incidence matrix of the underlying graph.
        """
        self.number_of_players = graph_matrix.shape[0]
        self._graph_matrix = np.copy(graph_matrix)
        _polish_graph_matrix(self._graph_matrix)

    def __repr__(self) -> str:  # pragma: no cover
        """Representation of icg."""
        return f"GraphGame({self._graph_matrix})"

    def get_value(self, coalition: Coalition) -> Value:
        """Get a value for coalition."""
        players = coalition.players
        value: Value = Value(0.0)
        for i, j in combinations(players, 2):
            value += self._graph_matrix[i, j]
        return value

    def get_values(self, coalitions: Iterable[Coalition] | None = None) -> np.ndarray[Any, np.dtype[Value]]:
        """Get values, or `False` if not known."""
        coalitions = coalitions if coalitions is not None else all_coalitions(self)
        return np.fromiter(map(self.get_value, coalitions), dtype=Value)

    def __eq__(self, other) -> bool:
        """Compare two games."""
        if isinstance(other, GraphCooperativeGame):
            return self.number_of_players == other.number_of_players and \
                bool(np.all(self._graph_matrix == other._graph_matrix))
        elif isinstance(other, Game):
            return self.number_of_players == other.number_of_players and \
                bool(np.all(self.get_values() == other.get_values()))
        raise AttributeError("Cannot compare games with anything else than games.")
