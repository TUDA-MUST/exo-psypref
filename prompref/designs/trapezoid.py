"""Ramella / Manzoori's six-parameter semi-trapezoid, as a Design.

This is the sibling project's family, reused rather than reimplemented so that the C1
baseline in an earlier step can be checked bit-for-bit against published numbers.
"""

from __future__ import annotations

import numpy as np
from preftune.profile import FEATURE_HIGH, FEATURE_LOW, FEATURE_NAMES, TorqueProfile

from .base import BoxDesign


class TrapezoidDesign(BoxDesign):
    """Extension and flexion bouts, each with peak torque, peak time and rise time."""

    param_names = tuple(FEATURE_NAMES)

    def __init__(self, stride_s: float | None = None) -> None:
        super().__init__(FEATURE_LOW, FEATURE_HIGH)
        self.stride_s = stride_s

    def curve(self, theta: np.ndarray) -> np.ndarray:
        return TorqueProfile(np.asarray(theta, dtype=float)).torque(stride_s=self.stride_s)

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Delegate to ``preftune``'s own batch sampler rather than ``BoxDesign``'s.

        ``BoxDesign.sample_prior`` draws plain uniforms, but
        ``preftune.experiment.run.make_batch`` rounds every feature to
        ``FEATURE_STEP`` ("the features were discretized with decimal precision").
        an earlier step's Phase 2 gate needs the two learners built on the *identical* batch,
        so this family's prior must apply the same rounding.
        """
        from preftune.profile import sample_batch

        return sample_batch(n, rng)
