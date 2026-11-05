"""A wearer whose preferred curve moves during the session.

The study's other cells assume a preference that holds still for 150 comparisons. Human
studies will not grant that, so this module lets the ideal move: a smooth time shift,
growing linearly over the session, sized to a total RMS displacement in Nm. Neither learner
A nor G models drift, so a drift cell measures a tolerance, not a success rate.

Most users' 101-point curves are **not** periodic: point 100 differs from point 0 by up to
0.49 Nm, because the curves are stride means. The shift therefore wraps the part that
leaves the cycle round to its start, and moves the existing seam to ``t = shift`` instead
of creating a new one; a zero shift returns the curve exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .oracle import Observer, rms_distance

#: The largest shift, in % of the gait cycle, that a drift may use.
MAX_SHIFT_PCT = 25.0

_T = np.arange(101.0)


def time_shift(curve, shift_pct: float) -> np.ndarray:
    """``curve`` delayed by ``shift_pct`` % of the gait cycle (positive means later
    assistance), linearly interpolated, wrapping round the cycle."""
    curve = np.asarray(curve, dtype=float)
    if curve.shape != (101,):
        raise ValueError(f"a user curve has 101 points, got shape {curve.shape}")
    shift = float(shift_pct)
    if not 0.0 <= shift < 100.0:
        raise ValueError(f"shift_pct must lie in [0, 100), got {shift}")
    source = _T - shift
    source = np.where(source < 0.0, source + 100.0, source)
    return np.interp(source, _T, curve)


def displacement(curve, shift_pct: float) -> float:
    """RMS distance in Nm between ``curve`` and its shifted copy."""
    return rms_distance(time_shift(curve, shift_pct), curve)


def shift_for_displacement(curve, target_nm: float, max_shift: float = MAX_SHIFT_PCT,
                           step: float = 0.25, tol: float = 1e-9) -> tuple[float, bool]:
    """The smallest shift in ``[0, max_shift]`` whose displacement reaches ``target_nm``,
    and whether it had to be capped.

    Displacement is continuous in the shift but not monotone over a whole cycle, so the
    first crossing is bracketed on a ``step`` grid and then bisected. When no shift up to
    ``max_shift`` reaches the target, the drift is capped at ``max_shift`` and flagged.
    """
    if target_nm <= 0.0:
        raise ValueError(f"target_nm must be positive, got {target_nm}")
    grid = np.arange(0.0, max_shift + step / 2.0, step)
    values = np.array([displacement(curve, shift) for shift in grid])
    above = np.flatnonzero(values >= target_nm)
    if above.size == 0:
        return float(max_shift), True
    low, high = float(grid[above[0] - 1]), float(grid[above[0]])
    while high - low > tol:
        mid = 0.5 * (low + high)
        if displacement(curve, mid) >= target_nm:
            high = mid
        else:
            low = mid
    return high, False


def path_for_shift(curve, shift_pct: float, n: int) -> np.ndarray:
    """The ideal at each of ``n`` comparisons: from the curve itself to its full shift."""
    return np.array([time_shift(curve, s) for s in np.linspace(0.0, float(shift_pct), int(n))])


def scale_for_displacement(curve, target_nm: float) -> float:
    """The growth ``g`` for which ``curve * (1 + g)`` lies ``target_nm`` (RMS) from ``curve``: the displacement of a scaling is ``g * rms(curve)``, so ``g`` is exact."""
    if target_nm <= 0.0:
        raise ValueError(f"target_nm must be positive, got {target_nm}")
    size = rms_distance(curve, np.zeros(np.shape(curve)))
    if size <= 0.0:
        raise ValueError("a zero curve cannot be scaled to a displacement")
    return float(target_nm) / size


def path_for_scale(curve, g: float, n: int) -> np.ndarray:
    """The ideal at each of ``n`` comparisons: ``curve * (1 + g * t / (n - 1))``."""
    curve = np.asarray(curve, dtype=float)
    growth = float(g) * np.linspace(0.0, 1.0, int(n))
    return curve[None, :] * (1.0 + growth)[:, None]


@dataclass
class DriftingObserver(Observer):
    """An :class:`Observer` that answers comparison ``k`` against ``ideal_path[k]``.

    After an answer, ``ideal_nm`` is the ideal that answer was given against, so the
    session loop -- which scores the learner's best curve against ``observer.ideal_nm``
    right after each answer -- measures every comparison against the preference current at
    that moment. Past the end of the path the last ideal holds.
    """

    ideal_path: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.ideal_path is None:
            raise ValueError("a DriftingObserver needs ideal_path")
        path = np.asarray(self.ideal_path, dtype=float)
        if path.ndim != 2 or path.shape[0] < 1:
            raise ValueError(f"ideal_path must be (n, points), got shape {path.shape}")
        if self.familiarity_weight:
            raise ValueError("drift and familiarity are not combined: familiarity moves the "
                             "target once, drift moves it every comparison")
        self.ideal_path = path
        self.ideal_nm = path[0]
        super().__post_init__()
        self._step = 0

    def respond(self, curve_a, curve_b, rng, order: int = 1) -> int:
        current = self.ideal_path[min(self._step, len(self.ideal_path) - 1)]
        self.ideal_nm = current
        self.effective_ideal_nm = current
        self._step += 1
        return super().respond(curve_a, curve_b, rng, order=order)
