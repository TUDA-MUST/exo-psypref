"""The Franks et al. (2021) hip torque parametrisation, as a Design.

Franks, Bryan, Martin, Reyes, Lakmazaheri & Collins (2021), Wearable Technologies 2:e16,
"Controller Parameterization": "Hip extension was defined by the rise time, peak time and peak
magnitude while hip flexion was defined by the peak time, peak magnitude and fall time. There
was a period of no torque in between the two peaks defined by the midpoint timing and its
duration." The curve is "a spline (piecewise cubic hermite interpolating polynomial) anchored
by nodes" -- scipy's PchipInterpolator.

Learners work in eight internal coordinates that keep the nodes ordered with no
repair, and in which every coordinate moves the curve:

- ``t_ext``, the extension peak time;
- ``log_gap_end`` and ``log_zero_dur``, the logs of the two no-torque periods in %GC (from the
  end of flexion to the next extension onset, and between the two bouts), each at least
  ``MIN_GAP`` = 0.5 %GC;
- ``lr_gap_ext``, ``lr_gap_flex``, ``lr_flex_fall``: the four torque-bearing durations (rise,
  extension peak to zero start, zero end to flexion peak, fall) are their minimums
  ``MIN_TORQUE`` plus a share of what the two no-torque periods leave, split by
  ``softmax([0, lr_gap_ext, lr_gap_flex, lr_flex_fall])``, so each ``lr`` is the log of a
  duration's excess over its minimum relative to the rise's;
- ``log_ext_peak``, ``log_flex_peak``.

``franks_parameters`` and ``internal_from_franks`` map the box one-to-one to and from Franks'
own eight. Units: Nm, % gait cycle, flexion positive (extension torque is negative).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from .base import BoxDesign

PHASE = np.linspace(0.0, 100.0, 101)

#: Internal coordinates.
PARAM_NAMES = ("t_ext", "log_gap_end", "log_zero_dur", "lr_gap_ext", "lr_gap_flex",
               "lr_flex_fall", "log_ext_peak", "log_flex_peak")
#: Franks' reported parameters.
FRANKS_NAMES = ("ext_rise", "ext_peak_time", "ext_peak", "zero_mid", "zero_dur",
                "flex_peak_time", "flex_peak", "flex_fall")

#: The shortest no-torque period, %GC, for both the end gap and the zero period: half a grid
#: step, so their two bounding nodes stay distinct.
MIN_GAP = 0.5
MIN_GAP_END = MIN_GAP
#: Minimums of rise, extension-to-zero gap, zero-to-flexion gap and fall, %GC (the old fit box's
#: lower bounds), so a torque bout always spans at least 1.5 %GC and covers a grid point.
MIN_TORQUE = np.array([1.0, 0.5, 0.5, 1.0])
#: Fit-box tops of the two no-torque periods. Together they leave at least
#: 100 - 60 - 30 - sum(MIN_TORQUE) = 7 %GC for the torque-bearing durations' excess.
MAX_GAP_END = 60.0
MAX_ZERO_DUR = 30.0
#: Each torque-bearing duration's excess lies within this factor of the rise's: the widest round
#: bound at which a +-0.05 step in every coordinate moves the curve (> 1e-6 Nm RMS) at every
#: point and corner of the fit box (log 15 passes at 1.6e-6; 10 leaves a 3x margin).
MAX_RATIO = 10.0
#: The extension peak time spans one full cycle, cut at 75 %GC (= -25), in the arc the fits leave
#: nearly empty (their 0.5-99.5th percentiles span -1..64 %GC; at most 0.1 % reach 75), so every
#: extension peak time has exactly one representative.
T_EXT_RANGE = (-25.0, 75.0)

#: Wide bounds, used only to fit the prior; the learner box is derived from the fits.
FIT_LOW = np.array([T_EXT_RANGE[0], np.log(MIN_GAP), np.log(MIN_GAP)]
                   + [-np.log(MAX_RATIO)] * 3 + [np.log(0.05), np.log(0.05)])
FIT_HIGH = np.array([T_EXT_RANGE[1], np.log(MAX_GAP_END), np.log(MAX_ZERO_DUR)]
                    + [np.log(MAX_RATIO)] * 3 + [np.log(30.0), np.log(30.0)])


def durations(theta) -> np.ndarray:
    """Rise, extension-to-zero gap, zero duration, zero-to-flexion gap and fall, in %GC. They
    sum to 100 %GC less the end gap, always: no repair is needed or applied."""
    th = np.asarray(theta, dtype=float)
    zero = np.exp(th[2])
    free = 100.0 - np.exp(th[1]) - zero - MIN_TORQUE.sum()
    logits = np.concatenate([[0.0], th[3:6]])
    w = np.exp(logits - logits.max())
    torque = MIN_TORQUE + free * w / w.sum()
    return np.array([torque[0], torque[1], zero, torque[2], torque[3]])


def nodes(theta) -> tuple[np.ndarray, np.ndarray]:
    """Seven nodes, unwrapped from the extension onset: onset, extension peak, zero start,
    zero end, flexion peak, flexion end, and the next onset (closing the cycle)."""
    th = np.asarray(theta, dtype=float)
    d = durations(th)
    onset = th[0] - d[0]
    times = onset + np.concatenate([[0.0], np.cumsum(d), [100.0]])
    values = np.array([0.0, -np.exp(th[6]), 0.0, 0.0, np.exp(th[7]), 0.0, 0.0])
    return times, values


def franks_parameters(theta) -> dict:
    th = np.asarray(theta, dtype=float)
    times, _ = nodes(th)
    d = durations(th)
    return {"ext_rise": float(d[0]), "ext_peak_time": float(th[0]),
            "ext_peak": float(np.exp(th[6])), "zero_mid": float(times[2] + d[2] / 2.0),
            "zero_dur": float(d[2]), "flex_peak_time": float(times[4]),
            "flex_peak": float(np.exp(th[7])), "flex_fall": float(d[4])}


def _internal(t_ext: float, d, ext_peak: float, flex_peak: float) -> np.ndarray:
    """Internal coordinates of the curve with extension peak ``t_ext``, the five durations
    ``d`` (%GC) and these peak magnitudes (Nm)."""
    d = np.asarray(d, dtype=float)
    gap_end = 100.0 - float(d.sum())
    excess = d[[0, 1, 3, 4]] - MIN_TORQUE
    if np.any(excess <= 0.0) or d[2] <= 0.0 or gap_end <= 0.0 or ext_peak <= 0.0 or flex_peak <= 0.0:
        raise ValueError(f"cannot be represented: durations {np.round(d, 4).tolist()} %GC (rise, "
                         f"gaps and fall must exceed {MIN_TORQUE.tolist()}), end gap "
                         f"{gap_end:.4f} %GC, peaks {ext_peak}, {flex_peak} Nm")
    return np.concatenate([[float(t_ext), np.log(gap_end), np.log(d[2])],
                           np.log(excess[1:] / excess[0]), [np.log(ext_peak), np.log(flex_peak)]])


def internal_from_franks(p: dict) -> np.ndarray:
    """The inverse of :func:`franks_parameters`: Franks' eight to internal coordinates."""
    zero_start = p["zero_mid"] - p["zero_dur"] / 2.0
    d = [p["ext_rise"], zero_start - p["ext_peak_time"], p["zero_dur"],
         p["flex_peak_time"] - (zero_start + p["zero_dur"]), p["flex_fall"]]
    return _internal(p["ext_peak_time"], d, p["ext_peak"], p["flex_peak"])


class FranksDesign(BoxDesign):
    """Franks' hip family in internal coordinates, inside a box (fit box by default)."""

    param_names = PARAM_NAMES

    def __init__(self, low=FIT_LOW, high=FIT_HIGH, prior=None) -> None:
        super().__init__(np.asarray(low, dtype=float), np.asarray(high, dtype=float))
        self.prior = prior

    def curve(self, theta: np.ndarray) -> np.ndarray:
        times, values = nodes(theta)
        # the previous cycle's flexion end closes the onset's left side: without it, scipy's end
        # condition gives the onset a steep slope, while Franks' curves start flat like every node
        spline = PchipInterpolator(np.r_[times[5] - 100.0, times], np.r_[0.0, values])
        local = np.mod(PHASE - times[0], 100.0) + times[0]
        return spline(local)

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.prior is None:
            raise ValueError("FranksDesign has no prior; fit one with prompref.priors.franks_prior")
        mu, sigma = self.prior
        return np.array([self.project(t) for t in rng.multivariate_normal(mu, sigma, size=n)])

    def _peak_time(self, phase: float) -> float:
        """The representative of ``phase`` (mod 100) inside the box's t_ext range, or nearest it."""
        lo, hi = float(self.low[0]), float(self.high[0])
        candidates = phase + np.array([-100.0, 0.0, 100.0])
        outside = np.maximum(lo - candidates, 0.0) + np.maximum(candidates - hi, 0.0)
        return float(candidates[int(np.argmin(outside))])

    def initial_guess(self, target: np.ndarray) -> np.ndarray:
        """Peaks read off the target; the gap between them split 0.3 / 0.2 / 0.5 into the
        extension-to-zero gap, the zero period and the zero-to-flexion gap; rise and fall of
        15 %GC each, shortened where the peaks lie so far apart that at least 1 %GC must stay
        free before the next onset."""
        c = np.asarray(target, dtype=float)[:-1]
        i_ext, i_flex = int(np.argmin(c)), int(np.argmax(c))
        t_ext = self._peak_time(float(PHASE[i_ext]))
        gap = float(np.clip(np.mod(PHASE[i_flex] - t_ext, 100.0), 3.0, 96.0))
        edge = float(np.clip((100.0 - gap - 1.0) / 2.0, 1.5, 15.0))
        d = [edge, 0.3 * gap, 0.2 * gap, 0.5 * gap, edge]
        guess = _internal(t_ext, d, max(-c[i_ext], 0.1), max(c[i_flex], 0.1))
        return self.project(guess)
