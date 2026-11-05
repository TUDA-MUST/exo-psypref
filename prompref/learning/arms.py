"""The Phase 5 prior arms.

All arms of a fold and family share one whitening and differ only in prior density and
search radius. The radius holds 99 % of the arm's prior mass:
- a Gaussian: the chi quantile, times ``sqrt(scale)``;
- a multivariate t, whose ``|z|^2 / d`` follows ``F(d, nu)``: ``sqrt(d F^-1_0.99(d, nu))``.

The wide arm is ``N(0, 9 I)``, three times the tight radius.

Gated: both gate scripts build their priors and duel radii here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import chi
from scipy.stats import f as f_dist

from .smc import BowlPrior

QUANTILE = 0.99


@dataclass(frozen=True)
class Arm:
    name: str
    nu: float | None = None
    scale: float = 1.0

    def radius(self, d: int, quantile: float = QUANTILE) -> float:
        if d < 1:
            raise ValueError(f"d must be at least 1, got {d}")
        if self.nu is None:
            return float(np.sqrt(self.scale) * chi.ppf(quantile, d))
        return float(np.sqrt(d * f_dist.ppf(quantile, d, self.nu)))

    def bowl_prior(self, base: BowlPrior = BowlPrior()) -> BowlPrior:
        """``base`` with this arm's z* prior; the ell and lambda priors stay ``base``'s (P10)."""
        return replace(base, nu=self.nu, scale=self.scale)


ARMS: dict[str, Arm] = {arm.name: arm for arm in (
    Arm("tight"),
    Arm("t8", nu=8.0),
    Arm("t4", nu=4.0),
    Arm("t2", nu=2.0),
    Arm("wide", scale=9.0),
)}
