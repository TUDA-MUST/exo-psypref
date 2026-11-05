"""Two split Gaussians on the circular gait cycle.

An extension part (negative) and a flexion part (positive), each with an amplitude, a
peak time, and separate rise and fall widths -- so a fast rise with a slow release is
expressible. Parameters are transformed so a Gaussian or Student-t prior over them makes
sense: log amplitudes, log widths, and peak times as offsets from population reference
centres. The references are the Molinaro biological peaks (about 5 and 54 %GC) moved by
the 100 ms centre shift of (about 9.5 %GC).

Each bump is summed over the neighbouring periods rather than wrapped with a modulo:
wrapping a split Gaussian leaves a jump at the antipode whenever rise and fall differ.

a later fix: the sum uses five periods, not three.
Truncating to three period copies means the sums at phase 0 and phase 100 differ by one
dropped copy each, whose distance can be as small as ~85 %GC at the box's widest width
(25 %GC), giving a wrap defect up to ~3e-3 Nm per unit of amplitude -- not negligible.
Wrapping the centre into [0, 100) bounds but does not eliminate this (the box's own
offset range reaches the seam exactly). Five periods push the nearest dropped copy to
~185 %GC away, where its contribution is at most ~5e-11 Nm at the box's most extreme
amplitude and width -- negligible, so the sum is exact for practical purposes.
"""

from __future__ import annotations

import numpy as np
from preftune.profile import PHASE_GRID
from scipy.optimize import least_squares

from .base import BoxDesign

PARAM_NAMES = ("log_amp_ext", "t_ext", "log_rise_ext", "log_fall_ext",
               "log_amp_flex", "t_flex", "log_rise_flex", "log_fall_flex")

EXT_REF_PCT = 15.0
FLEX_REF_PCT = 63.0

#: The shift centre the references above were read at, ms.
REFERENCE_SHIFT_MS = 100.0

#: Molinaro's mean stride time at 1.1 m/s, ms: 1.062 s over the ten AB30-AB34 legs, from
#: the Schaefer data report (out/schaefer/REPORT.md, section 6 "Timing relative to
#: biology", speed table). It converts a shift centre in ms to %GC on the Molinaro clock.
MOLINARO_STRIDE_MS = 1062.0


def reference_centres(shift_centre_ms: float) -> dict[str, float]:
    """``ext_ref``/``flex_ref`` moved with the population's shift centre.

    A Schaefer-centred population (138 ms) sits about 3.6 %GC later than the references,
    and its +250 ms stress users would pin peak offsets at the box edge; moving the
    references keeps offsets centred. At the R21 centre this returns the defaults exactly.
    """
    move = (float(shift_centre_ms) - REFERENCE_SHIFT_MS) / MOLINARO_STRIDE_MS * 100.0
    return dict(ext_ref=EXT_REF_PCT + move, flex_ref=FLEX_REF_PCT + move)

_LOG_AMP = (np.log(0.05), np.log(30.0))
_OFFSET = (-30.0, 30.0)
_LOG_WIDTH = (np.log(1.5), np.log(25.0))
LOW = np.array([_LOG_AMP[0], _OFFSET[0], _LOG_WIDTH[0], _LOG_WIDTH[0]] * 2)
HIGH = np.array([_LOG_AMP[1], _OFFSET[1], _LOG_WIDTH[1], _LOG_WIDTH[1]] * 2)


def split_bump(phase: np.ndarray, centre: float, rise: float, fall: float) -> np.ndarray:
    """Unit-height split Gaussian on the 100 %GC circle, smooth across the boundary.

    Summed over five period copies (-200, -100, 0, 100, 200), not three: a three-period
    sum drops one copy at phase 0 and a different one at phase 100, and at the box's
    widest width (25 %GC) that dropped copy can sit as close as ~85 %GC away -- not far
    enough to be negligible, so the curve is only approximately periodic. Five periods
    push the nearest dropped copy to ~185 %GC away, negligible at every width and
    amplitude the box allows.
    """
    phase = np.asarray(phase, dtype=float)
    total = np.zeros_like(phase)
    for period in (-200.0, -100.0, 0.0, 100.0, 200.0):
        dist = phase - centre + period
        width = np.where(dist < 0.0, rise, fall)
        total += np.exp(-0.5 * (dist / width) ** 2)
    return total


class DualGaussDesign(BoxDesign):
    def __init__(self, ext_ref: float = EXT_REF_PCT, flex_ref: float = FLEX_REF_PCT,
                 prior: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        super().__init__(LOW, HIGH)
        self.param_names = PARAM_NAMES
        self.ext_ref, self.flex_ref = float(ext_ref), float(flex_ref)
        self.prior = prior

    def curve(self, theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=float)
        if th.shape != (8,):
            raise ValueError(f"expected 8 parameters, got shape {th.shape}")
        ext = np.exp(th[0]) * split_bump(PHASE_GRID, self.ext_ref + th[1], np.exp(th[2]), np.exp(th[3]))
        flex = np.exp(th[4]) * split_bump(PHASE_GRID, self.flex_ref + th[5], np.exp(th[6]), np.exp(th[7]))
        return flex - ext

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.prior is None:
            raise ValueError("DualGaussDesign has no prior; fit one with "
                             "prompref.priors.data_prior")
        mu, sigma = self.prior
        return self.project(rng.multivariate_normal(mu, sigma, size=n))

    def initial_guess(self, target: np.ndarray) -> np.ndarray:
        """Amplitudes and peak offsets read off the target; 10 %GC widths."""
        c = np.asarray(target, dtype=float)
        i_ext, i_flex = int(np.argmin(c[:-1])), int(np.argmax(c[:-1]))

        def offset(index: int, ref: float) -> float:
            return float((PHASE_GRID[index] - ref + 50.0) % 100.0 - 50.0)

        guess = [np.log(max(-c[i_ext], 0.06)), offset(i_ext, self.ext_ref), np.log(10.0), np.log(10.0),
                 np.log(max(c[i_flex], 0.06)), offset(i_flex, self.flex_ref), np.log(10.0), np.log(10.0)]
        return self.project(np.array(guess))

    def fit_curve(self, target: np.ndarray, rng: np.random.Generator, n_starts: int = 8
                  ) -> tuple[np.ndarray, float]:
        """Multi-start bounded least squares; returns the best parameters and RMS (Nm)."""
        target = np.asarray(target, dtype=float)
        starts = [self.initial_guess(target)]
        starts += list(rng.uniform(self.low, self.high, size=(max(n_starts - 1, 0), 8)))
        best_theta, best_rms = None, np.inf
        for start in starts:
            result = least_squares(lambda th: self.curve(th) - target, start,
                                   bounds=(self.low, self.high))
            rms = float(np.sqrt(np.mean(result.fun ** 2)))
            if rms < best_rms:
                best_theta, best_rms = result.x, rms
        return np.asarray(best_theta, dtype=float), best_rms
