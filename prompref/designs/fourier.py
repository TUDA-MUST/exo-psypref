"""Truncated Fourier series over the gait cycle -- linear in its parameters.

Its purpose is to break a confound. If a preference is a ideal-point model in curve space then in a
linear-in-parameters family it is exactly a quadratic ideal-point model in parameter space, and in a
nonlinear family it is not. Without a linear family built on a different basis in the
comparison, "flexible beats rigid" and "linear beats nonlinear" cannot be separated, and
the ProMP is both.

No DC term: a constant offset is a continuous torque with no assistive role, so every
member of this family integrates to zero over the cycle.

The prior is fitted, not the box. A uniform draw of coefficients on +/-12 Nm
is an oscillating curve of about 9.5 Nm RMS, which is what a search starting from it
delivered -- the defect PrompS0 once had. Four harmonics fitted to the trapezoid family
miss its curves by 0.91 Nm median, as PrompS0's eight weights miss them by 0.94, so the
two linear families differ in their basis and not in their size or their prior.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from preftune.profile import PHASE_GRID

from .base import BoxDesign

#: Amplitude bound per coefficient, Nm.
COEFF_BOUND = 12.0

#: Four harmonics give eight coefficients -- PrompS0's size.
N_HARMONICS = 4


def harmonic_matrix(n_harmonics: int) -> np.ndarray:
    """``(101, 2 n)``: columns ``cos(2 pi h phi), sin(2 pi h phi)`` for ``h = 1 .. n``."""
    phase = 2.0 * np.pi * PHASE_GRID / 100.0
    columns = []
    for h in range(1, n_harmonics + 1):
        columns.append(np.cos(h * phase))
        columns.append(np.sin(h * phase))
    return np.column_stack(columns)


@lru_cache(maxsize=8)
def _fit_coefficient_prior(n_harmonics: int) -> tuple[np.ndarray, np.ndarray]:
    """Moments of the least-squares coefficients of 20000 trapezoid curves.

    The same construction as PrompS0's ``Sigma_param``: sample the parametric family,
    project each curve, take the moments. Cached per process: every design built in a
    sweep worker wants the same one.
    """
    from .trapezoid import TrapezoidDesign

    trapezoid = TrapezoidDesign()
    theta = trapezoid.sample_prior(np.random.default_rng(0), 20000)
    curves = np.array([trapezoid.curve(t) for t in theta])
    coef, *_ = np.linalg.lstsq(harmonic_matrix(n_harmonics), curves.T, rcond=None)
    coef = coef.T
    mu, sigma = coef.mean(axis=0), np.cov(coef, rowvar=False)
    # The cache hands this same array out to every design built in a sweep worker.
    # Mutating one design's prior in place would corrupt every other design's.
    mu.flags.writeable = False
    sigma.flags.writeable = False
    return mu, sigma


class FourierDesign(BoxDesign):
    """``sum_h a_h cos(2 pi h phi) + b_h sin(2 pi h phi)``."""

    def __init__(self, n_harmonics: int = N_HARMONICS,
                 prior: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        if n_harmonics < 1:
            raise ValueError(f"n_harmonics must be at least 1, got {n_harmonics}")
        self.n_harmonics = int(n_harmonics)
        n = 2 * self.n_harmonics
        super().__init__(np.full(n, -COEFF_BOUND), np.full(n, COEFF_BOUND))
        self.param_names = tuple(
            f"{kind}{h}" for h in range(1, self.n_harmonics + 1) for kind in ("a", "b")
        )
        self.design = harmonic_matrix(self.n_harmonics)
        if prior is None:
            self.prior = _fit_coefficient_prior(self.n_harmonics)
        else:
            mu, sigma = (np.asarray(a, dtype=float) for a in prior)
            if mu.shape != (n,) or sigma.shape != (n, n):
                # A wrong-shaped explicit prior is a silent, complete, wrong design --
                # multivariate_normal broadcasts or raises somewhere downstream, far from
                # here. Refuse it at construction instead.
                raise ValueError(
                    f"prior must be (mu, sigma) with mu.shape == ({n},) and sigma.shape "
                    f"== ({n}, {n}); got mu.shape={mu.shape}, sigma.shape={sigma.shape}"
                )
            self.prior = (mu, sigma)

    def curve(self, theta: np.ndarray) -> np.ndarray:
        return self.design @ np.asarray(theta, dtype=float)

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw from the fitted ``N(mu, Sigma)``, clipped into the feasibility box."""
        mu, sigma = self.prior
        return self.project(rng.multivariate_normal(mu, sigma, size=n))
