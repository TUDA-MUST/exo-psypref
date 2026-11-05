"""The interface every learner satisfies.

Deliberately parameter-based rather than index-based: a continuous method proposes any
point in the design space, and a batch method returns the parameter vectors of the two
candidates it picked. That is what lets one harness drive both.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PreferenceLearner(Protocol):
    def propose(self) -> tuple[np.ndarray, np.ndarray]:
        """The next pair of design points to show the wearer."""

    def update(self, theta_a: np.ndarray, theta_b: np.ndarray, response: int) -> None:
        """Record an answer: 0 prefers A, 1 prefers B, 2 no preference."""

    def best(self) -> np.ndarray:
        """Currently preferred design point."""

    def posterior_summary(self) -> dict:
        """Diagnostics and health flags for the run record."""
