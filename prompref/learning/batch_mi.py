"""Ramella's learner, adapted to the Design interface.

This is C1 when the design is the semi-trapezoid family (the published baseline) and C2
when it is a ProMP design (which isolates whether the search space or the duel design is
what wins). The learner itself is imported unchanged from the sibling project, and the
Phase 2 gate is that wrapping it changes nothing.

The wrapped learner seeds numpy's *legacy global* generator, matching APReL. The adapter
therefore must not draw from that generator itself, or the query sequence shifts. Build
the candidate batch from an explicit ``np.random.default_rng(seed)`` -- a separate bit
stream -- and never call ``np.random.rand``, ``np.random.randint``, ``np.random.normal``,
``np.random.seed``, or any other legacy global function in this module.
"""

from __future__ import annotations

import numpy as np
from preftune.learning.numpy_core import NumpyLearner

from ..designs.base import Design


class BatchMILearner:
    """Fixed candidate batch, linear reward, mutual-information queries."""

    def __init__(self, design: Design, seed: int = 0, n_batch: int = 40,
                 acquisition: str = "mutual_information") -> None:
        self.design = design
        self.n_batch = int(n_batch)
        # A dedicated generator, so sampling the batch never touches the legacy global
        # stream that NumpyLearner walks.
        batch = design.sample_prior(np.random.default_rng(seed), self.n_batch)
        self.batch = np.asarray(batch, dtype=float)
        self.inner = NumpyLearner(self.batch, seed=seed, acquisition=acquisition)
        self.last_indices: tuple[int, int] | None = None

    def propose(self) -> tuple[np.ndarray, np.ndarray]:
        i, j = self.inner.query()
        self.last_indices = (int(i), int(j))
        return self.batch[i], self.batch[j]

    def update(self, theta_a: np.ndarray, theta_b: np.ndarray, response: int) -> None:
        if self.last_indices is None:
            raise RuntimeError("call propose() before update()")
        i, j = self.last_indices
        # A "no preference" answer carries no direction, so this learner ignores it --
        # it has no likelihood term for indifference.
        if response != 2:
            self.inner.update(i, j, int(response))
        else:
            self.inner._pending = None
        self.last_indices = None

    def best(self) -> np.ndarray:
        return self.batch[self.inner.best()]

    def posterior_summary(self) -> dict:
        return {
            "weights": self.inner.weights(),
            "posterior_std": float(np.mean(self.inner.samples.std(axis=0))),
            "n_answers": len(self.inner._dataset),
        }
