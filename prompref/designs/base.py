"""The one interface every profile family satisfies.

Learners are generic over ``Design`` and never learn which family they are in; oracles
see only curves. That is what lets a six-parameter semi-trapezoid and a sixteen-weight
ProMP be scored on the same scale.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Design(Protocol):
    n_params: int
    param_names: tuple[str, ...]

    def curve(self, theta: np.ndarray) -> np.ndarray:
        """Torque on the 101-point %GC grid, in Nm, flexion positive."""

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``(n, n_params)`` draws from this family's prior."""

    def project(self, theta: np.ndarray) -> np.ndarray:
        """Nearest feasible parameter vector -- box, sign and structure constraints."""


class BoxDesign:
    """Shared behaviour for families whose feasible set is an axis-aligned box."""

    param_names: tuple[str, ...] = ()

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        if self.low.shape != self.high.shape:
            raise ValueError("low and high must have the same shape")
        if np.any(self.high < self.low):
            raise ValueError("every high must be at least its low")

    @property
    def n_params(self) -> int:
        return int(self.low.size)

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=(n, self.n_params))

    def project(self, theta: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(theta, dtype=float), self.low, self.high)
