"""Schaefer et al.'s four-half-cosine hip profile with both peak magnitudes free.

our design notes, section 9a, "P. Interpretable families and how many parameters a
profile needs", family (ii): Schaefer et al.'s formulation with "the two
peak magnitudes free, 6 parameters". Family (i), as their study used it, is
:class:`~prompref.designs.schaefer.SchaeferDesign`: four timings, both peaks fixed at 12 Nm.

Parameters: the four timings t1..t4 of :class:`SchaeferDesign` (zero-crossing, extension peak,
zero-crossing, flexion peak, %GC), in its box and with its cycle-order repair, and the extension
and flexion peak magnitudes in Nm, each in [0, 12] (the device limit). The curve is
:class:`SchaeferDesign`'s with those peaks, so at peaks (12, 12) it is family (i)'s curve exactly,
and family (i) is the face of this box where both peaks sit on their upper bound.

Opt-in: nothing that existed before this module reads it.
"""

from __future__ import annotations

import numpy as np
from preftune.profile import PHASE_GRID

from ..population.curves import TORQUE_LIMIT_NM
from .base import BoxDesign
from .schaefer import PREFERRED_MEAN, TIMING_HIGH, TIMING_LOW, SchaeferDesign, _project_ordered

PARAM_NAMES = ("t1", "t2", "t3", "t4", "ext_peak", "flex_peak")
#: Peak magnitudes, Nm: from zero to the device limit (R29), which Schaefer's 12 Nm also is.
PEAK_LOW = 0.0
PEAK_HIGH = float(TORQUE_LIMIT_NM)
LOW = np.concatenate([TIMING_LOW, [PEAK_LOW, PEAK_LOW]])
HIGH = np.concatenate([TIMING_HIGH, [PEAK_HIGH, PEAK_HIGH]])


def representative(phase: float, low: float, high: float) -> float:
    """The representative of ``phase`` (mod 100) inside [low, high], or the one nearest it."""
    candidates = float(phase) + np.array([-100.0, 0.0, 100.0])
    outside = np.maximum(low - candidates, 0.0) + np.maximum(candidates - high, 0.0)
    return float(candidates[int(np.argmin(outside))])


def _crossing(c: np.ndarray, start: int, stop: int, step: int) -> int | None:
    """The first index from ``start`` toward ``stop`` (circularly, by ``step``) at which the
    sampled curve ``c`` (one cycle, no duplicated end point) is at or above zero."""
    n = c.size
    i = start
    while i % n != stop % n:
        i += step
        if c[i % n] >= 0.0:
            return i % n
    return None


class SchaeferPeaksDesign(BoxDesign):
    """Schaefer's four timings plus the extension and flexion peak magnitudes (6 parameters)."""

    param_names = PARAM_NAMES

    def __init__(self, prior: tuple[np.ndarray, np.ndarray] | None = None, low=None,
                 high=None) -> None:
        """``low`` and ``high`` replace the box; by default
        it is family (i)'s timing box with peaks in [0, 12] Nm."""
        super().__init__(LOW if low is None else np.asarray(low, dtype=float),
                         HIGH if high is None else np.asarray(high, dtype=float))
        #: The fitted moments (``SurfacePrior``), or ``None`` for the box-uniform default.
        self.prior = prior

    def curve(self, theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=float)
        return SchaeferDesign(peak_ext_nm=float(th[4]), peak_flex_nm=float(th[5])).curve(th[:4])

    def project(self, theta: np.ndarray) -> np.ndarray:
        """Timings as :class:`SchaeferDesign` projects them (box, then cycle order); peaks
        clipped into [0, 12] Nm."""
        th = np.asarray(theta, dtype=float)
        single = th.ndim == 1
        th2 = np.atleast_2d(th)
        timings = np.atleast_2d(_project_ordered(th2[:, :4], self.low[:4], self.high[:4]))
        peaks = np.clip(th2[:, 4:], self.low[4:], self.high[4:])
        out = np.concatenate([timings, peaks], axis=1)
        return out[0] if single else out

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """With ``prior``: N(mu, sigma), repaired through :meth:`project`. Without: family (i)'s
        box-uniform timings with order rejection and uniform peaks."""
        if self.prior is None:
            timings = SchaeferDesign().sample_prior(rng, n)
            peaks = rng.uniform(self.low[4:], self.high[4:], size=(int(n), 2))
            return np.concatenate([timings, peaks], axis=1)
        mu, sigma = self.prior
        return self.project(rng.multivariate_normal(mu, sigma, size=int(n)))

    def initial_guess(self, target: np.ndarray) -> np.ndarray:
        """Peaks and their timings read off the target (its minimum and maximum); t3 the first
        sample at or above zero after the extension peak, t1 the last one before it (searching
        no further than the flexion peak), each falling back to Schaefer's published preferred
        timing when the target never crosses zero there."""
        c = np.asarray(target, dtype=float)[:-1]
        phase = PHASE_GRID[:-1]
        i_ext, i_flex = int(np.argmin(c)), int(np.argmax(c))
        i3 = _crossing(c, i_ext, i_flex, +1)
        i1 = _crossing(c, i_ext, i_flex, -1)
        t1 = float(phase[i1]) if i1 is not None else float(PREFERRED_MEAN[0])
        t3 = float(phase[i3]) if i3 is not None else float(PREFERRED_MEAN[2])
        guess = np.array([representative(t1, self.low[0], self.high[0]),
                          representative(float(phase[i_ext]), self.low[1], self.high[1]),
                          representative(t3, self.low[2], self.high[2]),
                          representative(float(phase[i_flex]), self.low[3], self.high[3]),
                          max(-float(c[i_ext]), 0.0), max(float(c[i_flex]), 0.0)])
        return self.project(guess)
