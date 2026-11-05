"""The simulated user's true just-noticeable difference, and the variants R34 sweeps.

The observer's utility is linear in RMS curve distance, so a duel whose two curves differ by
``delta`` puts the probit argument at ``delta / s``. The conventional 75 %-correct threshold
is therefore ``Phi^-1(0.75) * s`` -- 0.3372 Nm at s = 0.5. Lapse is excluded: a lapse best-profile distances
performance everywhere and would make the "75 % point" depend on it, while the JND a
comparator is handed is meant to describe perception, not attention.
"""

from __future__ import annotations

from scipy.stats import norm

#: R34's levels. ``None`` is the gate turned off, not a JND of zero.
JND_SCALES: tuple[float | None, ...] = (0.5, 1.0, 2.0, None)


def true_jnd(s: float) -> float:
    """The 75 %-correct curve distance in Nm RMS for a simulated user of sensitivity ``s``."""
    if s <= 0.0:
        raise ValueError(f"s must be positive, got {s}")
    return float(norm.ppf(0.75)) * float(s)


def gate_value(s: float, scale: float | None) -> float | None:
    """The JND a comparator is given: ``scale`` times the true one, or ``None`` for off."""
    if scale is None:
        return None
    if scale <= 0.0:
        raise ValueError(f"scale must be positive or None, got {scale}")
    return float(scale) * true_jnd(s)
