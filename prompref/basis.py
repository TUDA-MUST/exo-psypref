"""Circular von Mises basis over the gait cycle, normalised to a partition of unity.

Normalising so that ``sum_k b_k(phi) = 1`` makes each weight ``w_k`` read as a local
torque value in Nm rather than an abstract coefficient, which keeps the prior's scale
interpretable and makes a constant curve exactly representable.

The concentration ``kappa`` is not hard-coded but derived from a target *overlap* -- the
height of one basis function at the midpoint between adjacent centres, relative to its
own peak. That inverse is closed-form, so no optimiser is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from preftune.profile import N_PHASE, PHASE_GRID

DEFAULT_N_BASIS = 8
DEFAULT_OVERLAP = 0.5


def kappa_for_overlap(n_basis: int, overlap: float = DEFAULT_OVERLAP) -> float:
    """Concentration giving ``overlap`` between adjacent basis functions.

    With the peak-shifted kernel ``exp(kappa (cos(2 pi d) - 1))`` the value at the
    midpoint between neighbours (``d = 1 / 2K``) relative to the peak is
    ``exp(kappa (cos(pi / K) - 1))``. Inverting gives the expression below.
    """
    if not 0.0 < overlap < 1.0:
        raise ValueError(f"overlap must be in (0, 1), got {overlap}")
    if n_basis < 2:
        raise ValueError(f"n_basis must be at least 2, got {n_basis}")
    return float(np.log(overlap) / (np.cos(np.pi / n_basis) - 1.0))


@dataclass(frozen=True)
class VonMisesBasis:
    """A partition-of-unity circular basis on the 101-point %GC grid."""

    n_basis: int = DEFAULT_N_BASIS
    overlap: float = DEFAULT_OVERLAP
    kappa: float = field(init=False)
    centres: np.ndarray = field(init=False)
    design: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kappa", kappa_for_overlap(self.n_basis, self.overlap))
        centres = np.arange(self.n_basis, dtype=float) * (100.0 / self.n_basis)
        object.__setattr__(self, "centres", centres)

        # Shift the exponent so its maximum is 0: exp then never overflows, and the
        # normalisation below is unaffected by the common factor.
        delta = (PHASE_GRID[:, None] - centres[None, :]) / 100.0
        raw = np.exp(self.kappa * (np.cos(2.0 * np.pi * delta) - 1.0))
        object.__setattr__(self, "design", raw / raw.sum(axis=1, keepdims=True))

    @property
    def gram(self) -> np.ndarray:
        """``Phi^T Phi / N`` -- the metric that turns weight distance into curve distance."""
        return self.design.T @ self.design / N_PHASE

    def curve(self, w: np.ndarray) -> np.ndarray:
        """Torque on the phase grid, in Nm."""
        w = np.asarray(w, dtype=float)
        if w.shape[-1] != self.n_basis:
            raise ValueError(f"expected {self.n_basis} weights, got shape {w.shape}")
        return self.design @ w

    def project(self, curve: np.ndarray) -> np.ndarray:
        """Least-squares weights for an arbitrary curve.

        Accepts a single ``(101,)`` curve or a ``(101, n)`` stack of them, because
        fitting a prior projects thousands of curves at once. The phase axis is always
        axis 0, which is what ``lstsq`` needs.
        """
        curve = np.asarray(curve, dtype=float)
        if curve.shape[0] != N_PHASE:
            raise ValueError(
                f"expected {N_PHASE} samples on axis 0, got shape {curve.shape}"
            )
        return np.linalg.lstsq(self.design, curve, rcond=None)[0]
