"""The interface a preference learner must satisfy.

Kept deliberately narrow so the APReL reference and the numpy core written in
Phase 3 are interchangeable inside :mod:`preftune.experiment.run`, and so an
equivalence test can drive both through the same calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def normalise_features(batch: np.ndarray) -> np.ndarray:
    """Z-score the six features across the batch.

    A reward linear in raw features would be dominated by whichever feature happens
    to carry the largest numbers; Ramella's timing features span tens of %GC while
    the torques span a few Nm.
    """
    batch = np.atleast_2d(np.asarray(batch, dtype=float))
    spread = batch.std(axis=0)
    spread[spread == 0.0] = 1.0
    return (batch - batch.mean(axis=0)) / spread


class Learner(ABC):
    """Learns a reward over a fixed batch of candidates from pairwise answers."""

    def __init__(self, batch: np.ndarray, seed: int = 0) -> None:
        self.batch = np.atleast_2d(np.asarray(batch, dtype=float))
        self.features = normalise_features(self.batch)
        self.seed = seed
        self.history: list[dict] = []

    @property
    def n_candidates(self) -> int:
        return self.batch.shape[0]

    @abstractmethod
    def query(self) -> tuple[int, int]:
        """Choose the next pair of candidate indices to show the user."""

    @abstractmethod
    def update(self, i: int, j: int, response: int) -> None:
        """Record an answer: ``response`` is ``0`` for candidate ``i``, ``1`` for ``j``."""

    @abstractmethod
    def weights(self) -> np.ndarray:
        """Posterior mean reward weights, unit-norm."""

    def rewards(self) -> np.ndarray:
        """Estimated reward of every candidate in the batch."""
        return self.features @ self.weights()

    def best(self) -> int:
        """The candidate the learner currently believes is preferred."""
        return int(np.argmax(self.rewards()))
