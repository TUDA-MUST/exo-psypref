"""Ramella et al.'s semi-trapezoid family as a projection surface.

The same six features, box and curve as :class:`~prompref.designs.trapezoid.TrapezoidDesign`
(C1's family; ``preftune.profile``, Manzoori et al. 2024 / Ramella et al. 2025 Table I), plus the
``initial_guess`` the projection's least-squares fitter needs (:class:`~prompref.learning.lens.
SurfaceFitter`). ``TrapezoidDesign`` itself is unchanged, so C1's sessions cannot move.
"""

from __future__ import annotations

import numpy as np
from preftune.profile import PHASE_GRID

from .schaefer_peaks import representative
from .trapezoid import TrapezoidDesign

#: The rise time the guess starts from, %GC: the middle of Ramella's 10-20 %GC range.
GUESS_RISE = 15.0


class TrapezoidSurface(TrapezoidDesign):
    """C1's semi-trapezoid with an initial guess. ``prior`` is accepted for the surface-prior
    interface and stored; sampling stays Ramella's own box sampler."""

    def __init__(self, prior: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        super().__init__()
        self.prior = prior

    def initial_guess(self, target: np.ndarray) -> np.ndarray:
        """Each part's peak torque and plateau centre read off the target's minimum (extension)
        and maximum (flexion), its rise time at the middle of the range, clipped into the box
        (peak times through their nearest representative mod 100)."""
        c = np.asarray(target, dtype=float)[:-1]
        phase = PHASE_GRID[:-1]
        i_ext, i_flex = int(np.argmin(c)), int(np.argmax(c))
        guess = np.array([-float(c[i_ext]),
                          representative(float(phase[i_ext]), self.low[1], self.high[1]),
                          GUESS_RISE,
                          float(c[i_flex]),
                          representative(float(phase[i_flex]), self.low[4], self.high[4]),
                          GUESS_RISE])
        return self.project(guess)
