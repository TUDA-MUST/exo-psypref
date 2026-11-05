"""Curve operations shared by the population generator and the priors.

Two clocks appear here: the native 200 Hz series of a Molinaro trial, and the 101-point
%GC grid. Torque is in Nm, flexion positive.
"""

from __future__ import annotations

import numpy as np
from preftune.profile import N_PHASE, PHASE_GRID

#: Peak torque the device can deliver, Nm.
TORQUE_LIMIT_NM = 12.0

#: The four landmarks, in cycle order from the extension peak.
LANDMARKS = ("ext_peak", "ext_to_flex", "flex_peak", "flex_to_ext")

#: Moved landmarks closer than this are treated as a change of order.
MIN_LANDMARK_GAP_PCT = 2.0

_GRID_STEP = 100.0 / (N_PHASE - 1)


def shift_signal(x: np.ndarray, shift_ms: float, fs: float) -> np.ndarray:
    """Shift later by ``shift_ms`` (earlier if negative), whole samples, holding the edge.

    ``preftune.data.loader.delay`` only delays; a personal timing shift can also be
    negative.
    """
    x = np.asarray(x, dtype=float)
    n = int(round(shift_ms * 1e-3 * fs))
    if abs(n) >= x.size:
        raise ValueError(f"a shift of {n} samples is longer than the signal ({x.size})")
    if n == 0:
        return x.copy()
    if n > 0:
        return np.concatenate([np.full(n, x[0]), x[:-n]])
    return np.concatenate([x[-n:], np.full(-n, x[-1])])


def stride_curves(signal: np.ndarray, gait_phase: np.ndarray, cycles, n: int = N_PHASE
                  ) -> np.ndarray:
    """One curve per heel-strike-to-heel-strike cycle on an ``n``-point %GC grid.

    On a raw per-stride cut, point 100 equals point 0 only approximately, because the two
    ends are adjacent heel strikes, not the same instant (measured up to about 0.2 Nm on
    real strides); ``warp_landmarks`` output is exactly periodic.
    """
    grid = np.linspace(0.0, 1.0, n)
    signal = np.asarray(signal, dtype=float)
    phase = np.asarray(gait_phase, dtype=float)
    return np.array([np.interp(grid, phase[a:b], signal[a:b]) for a, b in cycles])


def physics_clip(curve: np.ndarray, limit: float = TORQUE_LIMIT_NM) -> np.ndarray:
    """The device's peak-torque limit."""
    return np.clip(np.asarray(curve, dtype=float), -limit, limit)


def landmarks(curve: np.ndarray) -> np.ndarray:
    """Phases (%GC) of the extension peak, extension-to-flexion crossing, flexion peak and
    flexion-to-extension crossing.

    Peaks are the global minimum and maximum. A crossing is the *last* sign change on the
    forward arc between the two peaks, so a curve that dithers around zero before
    committing is placed where it commits (three Molinaro legs dither). Crossings are
    linearly interpolated between grid points.
    """
    c = np.asarray(curve, dtype=float)
    if c.shape != (N_PHASE,):
        raise ValueError(f"expected a {N_PHASE}-point curve, got shape {c.shape}")
    m = N_PHASE - 1  # distinct points on the circle; point 100 duplicates point 0
    i_ext, i_flex = int(np.argmin(c[:m])), int(np.argmax(c[:m]))
    if c[i_ext] >= 0.0 or c[i_flex] <= 0.0:
        raise ValueError("the curve needs both an extension (negative) and a flexion "
                         "(positive) bout")

    def last_crossing(start: int, stop: int, rising: bool) -> float:
        found = None
        idx = start
        while idx != stop:
            nxt = (idx + 1) % m
            a, b = c[idx], c[nxt]
            if (rising and a < 0.0 <= b) or (not rising and a > 0.0 >= b):
                found = (PHASE_GRID[idx] + (a / (a - b)) * _GRID_STEP) % 100.0
            idx = nxt
        if found is None:
            kind = "extension-to-flexion" if rising else "flexion-to-extension"
            raise ValueError(f"no {kind} crossing between the peaks")
        return float(found)

    return np.array([PHASE_GRID[i_ext], last_crossing(i_ext, i_flex, True),
                     PHASE_GRID[i_flex], last_crossing(i_flex, i_ext, False)])


def warp_landmarks(curve: np.ndarray, offsets) -> np.ndarray:
    """Move the four landmarks by ``offsets`` (%GC), keeping the curve between them.

    A monotone piecewise-linear circular time map sends each landmark ``L_i`` to
    ``L_i + offsets[i]``; the new curve at phase ``p`` is the old curve at the pre-image
    of ``p``. Offsets that would change the landmark order, or leave two landmarks at or
    within 2 %GC of each other, raise ``ValueError``.
    """
    c = np.asarray(curve, dtype=float)
    off = np.asarray(offsets, dtype=float)
    if off.shape != (len(LANDMARKS),):
        raise ValueError("offsets need one entry per landmark")
    src = landmarks(c)
    gaps = (np.roll(src, -1) - src) % 100.0
    s = src[0] + np.concatenate([[0.0], np.cumsum(gaps[:3])])
    d = s + off
    d_knots = np.append(d, d[0] + 100.0)
    if np.any(np.diff(d_knots) <= MIN_LANDMARK_GAP_PCT):
        raise ValueError("offsets would change the landmark order, or bring two landmarks "
                         f"within {MIN_LANDMARK_GAP_PCT} %GC")
    s_knots = np.append(s, s[0] + 100.0)
    target = d_knots[0] + (PHASE_GRID - d_knots[0]) % 100.0
    source = np.interp(target, d_knots, s_knots) % 100.0
    return np.interp(source, PHASE_GRID, c)
