"""Two circular bumps: amplitude, centre and width for each of the two parts.

The simplest smooth two-part family, and the reference that the structured ProMP (S2)
must reproduce exactly when its within-part shape weights are zero.

Sign convention: flexion positive, extension negative -- the extension bump is
subtracted, the flexion bump is added.
"""

from __future__ import annotations

import numpy as np
from preftune.profile import PHASE_GRID

from .base import BoxDesign

LOW = np.array([4.0, 5.0, 10.0, 4.0, 50.0, 10.0])
HIGH = np.array([14.0, 35.0, 45.0, 14.0, 80.0, 45.0])


def circular_bump(centre_pct: float, fwhm_pct: float) -> np.ndarray:
    """Unit-height circular bump, ``fwhm_pct`` wide at half maximum."""
    # Half maximum at +/- fwhm/2 fixes the concentration, by the same inverse as the
    # basis: exp(kappa (cos(2 pi d) - 1)) = 0.5 at d = fwhm / 200.
    kappa = np.log(0.5) / (np.cos(np.pi * fwhm_pct / 100.0) - 1.0)
    delta = (PHASE_GRID - centre_pct) / 100.0
    return np.exp(kappa * (np.cos(2.0 * np.pi * delta) - 1.0))


class BumpDesign(BoxDesign):
    """Extension bump (negative) plus flexion bump (positive)."""

    param_names = (
        "ext_amp", "ext_centre", "ext_width",
        "flex_amp", "flex_centre", "flex_width",
    )

    def __init__(self) -> None:
        super().__init__(LOW, HIGH)

    def curve(self, theta: np.ndarray) -> np.ndarray:
        ext_a, ext_c, ext_w, flex_a, flex_c, flex_w = np.asarray(theta, dtype=float)
        return (
            -ext_a * circular_bump(ext_c, ext_w)
            + flex_a * circular_bump(flex_c, flex_w)
        )
