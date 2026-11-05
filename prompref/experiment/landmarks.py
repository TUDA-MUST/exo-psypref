"""Whether a curve has the structure a preferred assistance curve must have (Phase 5a M8).

Moved here from ``scripts/phase5_prior_diagnostics.py`` so the Phase 5 harness can report
it for every delivered curve; the script imports it back unchanged.
"""

from __future__ import annotations

import numpy as np

from ..population.curves import MIN_LANDMARK_GAP_PCT, landmarks


def two_bout_plausible(curve: np.ndarray) -> bool:
    """Both bouts exist and the four landmarks keep their circular order, every gap above
    ``MIN_LANDMARK_GAP_PCT`` -- the structure a preferred assistance curve must have (M8)."""
    try:
        found = landmarks(curve)
    except ValueError:
        return False
    gaps = (np.roll(found, -1) - found) % 100.0
    return bool(abs(gaps.sum() - 100.0) < 1e-6 and np.all(gaps > MIN_LANDMARK_GAP_PCT))
