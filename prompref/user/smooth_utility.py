"""A random smooth utility over curves.

The simulated user's utility of a curve tau is

    u(tau) = kappa * [ g(x(tau)) - ||(x(tau) - c) / rho||^2 / 2 ]

- ``x(tau)`` are the first six Fourier harmonics of the curve, 12 coefficients (cosine and
  sine of harmonics 1..6, in that order), read on the cycle's 100 distinct points (the 101st
  duplicates the first) and standardised per coefficient by the mean and SD of the fold's
  training ideals. This feature map is deliberately not A's ProMP coordinates.
- ``g`` is a sample of a zero-mean Gaussian process with an RBF kernel of unit variance and
  lengthscale :data:`LENGTHSCALE` in standardised units, drawn through
  :data:`N_FEATURES` random Fourier features (Rahimi & Recht 2007):
  ``g(x) = sqrt(2 / D) sum_j w_j cos(omega_j. x + b_j)`` with ``omega_j ~ N(0, I / l^2)``,
  ``b_j ~ U[0, 2 pi)`` and ``w_j ~ N(0, 1)``. Each user's draw is seeded from the answer seed
  with its own salt (:func:`draw_features`), so a session rebuilds it exactly.
- ``c`` is the user's centre, a draw from the population's feature distribution, and
  ``rho`` = :data:`RHO` SD is a weak envelope that keeps the optimum bounded.
- ``kappa`` is the user's scale: it matches the simulated user's decisiveness to the plain s = 1.0 RMS
  simulated user's, as the Phase 5e REPORT section 9 measures decisiveness (the median over
  prior-scale duel pairs of |du| / |du_rms|). ``scripts/icorr_robust_prep.py`` measures it and
  stores it with ``c`` in the fold's sidecar (:mod:`prompref.experiment.robust`).

The simulated user answers by a probit on du with its lapse, like every other observer
(:class:`prompref.user.simulated_user.Observer`, ``metric="random_smooth"``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from preftune.profile import N_PHASE, PHASE_GRID

#: Harmonics 1..6: 12 coefficients.
N_HARMONICS = 6
N_COEFFICIENTS = 2 * N_HARMONICS
#: The RBF lengthscale in standardised feature units.
LENGTHSCALE = 1.5
#: The envelope's width in SD.
RHO = 3.0
#: Random Fourier features per user. Not planned: chosen so the feature kernel matches the
#: RBF kernel to about 0.03 (1 / sqrt(D)) at every distance.
N_FEATURES = 2000
#: Salt of the per-user generator of g's features (the answer seed is the other half).
RFF_SALT = 0x125A0F
#: Salt of the per-user generator of the centre's draw.
CENTRE_SALT = 0x125C3E
#: Curves per block when many are evaluated at once (memory only; values are unchanged).
CHUNK = 2048


def feature_matrix() -> np.ndarray:
    """``(12, 101)``: row ``2 (h - 1)`` gives the cosine coefficient of harmonic ``h``, row
    ``2 (h - 1) + 1`` its sine coefficient, ``(2 / N) sum_n tau_n cos(2 pi h phi_n)`` over the
    ``N = 100`` distinct points. The duplicated last point gets weight 0."""
    n = N_PHASE - 1
    phase = 2.0 * np.pi * PHASE_GRID[:-1] / 100.0
    rows = []
    for h in range(1, N_HARMONICS + 1):
        rows.append(np.cos(h * phase) * (2.0 / n))
        rows.append(np.sin(h * phase) * (2.0 / n))
    matrix = np.zeros((N_COEFFICIENTS, N_PHASE))
    matrix[:, :-1] = np.array(rows)
    return matrix


_FEATURES = feature_matrix()


def fourier_coefficients(curves) -> np.ndarray:
    """The 12 raw Fourier coefficients of one ``(101,)`` curve or a ``(n, 101)`` stack."""
    curves = np.asarray(curves, dtype=float)
    return curves @ _FEATURES.T


def draw_features(answer_seed: int, n_features: int = N_FEATURES,
                  lengthscale: float = LENGTHSCALE):
    """``(omega, phase, weight)`` of one user's g, from ``default_rng([answer_seed, RFF_SALT])``:
    omega ``(D, 12)`` first, then the phases, then the weights."""
    rng = np.random.default_rng([int(answer_seed), RFF_SALT])
    omega = rng.standard_normal((int(n_features), N_COEFFICIENTS)) / float(lengthscale)
    phase = rng.uniform(0.0, 2.0 * np.pi, int(n_features))
    weight = rng.standard_normal(int(n_features))
    return omega, phase, weight


def draw_centre_index(answer_seed: int, n_population: int) -> int:
    """Which of the fold's ``n_population`` training ideals is the user's centre: uniform, from
    ``default_rng([answer_seed, CENTRE_SALT])``."""
    return int(np.random.default_rng([int(answer_seed), CENTRE_SALT]).integers(int(n_population)))


@dataclass(frozen=True, eq=False)
class RandomSmoothUtility:
    """``u(tau) = scale * [g(x(tau)) - ||(x(tau) - centre) / rho||^2 / 2]`` (module docstring).

    ``feature_mean`` and ``feature_sd`` standardise the raw coefficients; ``centre`` is in
    standardised units. ``scale`` is the matched-decisiveness factor kappa; 1.0 is the raw
    utility.
    """

    feature_mean: np.ndarray
    feature_sd: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    weight: np.ndarray
    centre: np.ndarray
    rho: float = RHO
    scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("feature_mean", "feature_sd", "omega", "phase", "weight", "centre"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        if self.feature_mean.shape != (N_COEFFICIENTS,) or self.feature_sd.shape != (N_COEFFICIENTS,):
            raise ValueError("feature_mean and feature_sd need 12 coefficients")
        if np.any(self.feature_sd <= 0.0):
            raise ValueError("feature_sd must be positive")
        if self.omega.ndim != 2 or self.omega.shape[1] != N_COEFFICIENTS:
            raise ValueError(f"omega must be (D, 12), got {self.omega.shape}")
        if self.phase.shape != (self.omega.shape[0],) or self.weight.shape != (self.omega.shape[0],):
            raise ValueError("phase and weight need one entry per Fourier feature")
        if self.centre.shape != (N_COEFFICIENTS,):
            raise ValueError("centre needs 12 coefficients")
        if not self.rho > 0.0 or not self.scale > 0.0:
            raise ValueError("rho and scale must be positive")

    @classmethod
    def for_user(cls, answer_seed: int, feature_mean, feature_sd, centre, *,
                 scale: float = 1.0, n_features: int = N_FEATURES,
                 lengthscale: float = LENGTHSCALE, rho: float = RHO) -> "RandomSmoothUtility":
        """The user's utility: g drawn from the answer seed, the rest given."""
        omega, phase, weight = draw_features(answer_seed, n_features, lengthscale)
        return cls(feature_mean, feature_sd, omega, phase, weight, centre, rho=rho, scale=scale)

    def with_scale(self, scale: float) -> "RandomSmoothUtility":
        return RandomSmoothUtility(self.feature_mean, self.feature_sd, self.omega, self.phase,
                                   self.weight, self.centre, rho=self.rho, scale=float(scale))

    def features(self, curves) -> np.ndarray:
        """Standardised features of one curve ``(12,)`` or a stack ``(n, 12)``."""
        return (fourier_coefficients(curves) - self.feature_mean) / self.feature_sd

    def gp(self, x) -> np.ndarray:
        """g at standardised features ``x``, one point ``(12,)`` or many ``(n, 12)``; many
        points in chunks of :data:`CHUNK`, so the ``(n, D)`` intermediate stays small."""
        x = np.asarray(x, dtype=float)
        amp = np.sqrt(2.0 / self.weight.size)
        if x.ndim == 1 or x.shape[0] <= CHUNK:
            return amp * (np.cos(x @ self.omega.T + self.phase) @ self.weight)
        return np.concatenate([amp * (np.cos(x[i:i + CHUNK] @ self.omega.T + self.phase)
                                      @ self.weight) for i in range(0, x.shape[0], CHUNK)])

    def raw(self, curves):
        """The unscaled utility of one curve (a float) or a stack (an array)."""
        x = self.features(curves)
        envelope = 0.5 * np.sum(((x - self.centre) / self.rho) ** 2, axis=-1)
        value = self.gp(x) - envelope
        return float(value) if np.ndim(value) == 0 else value

    def __call__(self, curve) -> float:
        return float(self.scale * self.raw(curve))

    def value_and_gradient(self, curve) -> tuple[float, np.ndarray]:
        """``(u, du / dcurve)`` of one ``(101,)`` curve, scaled; for the maximisation."""
        curve = np.asarray(curve, dtype=float)
        x = self.features(curve)
        arg = self.omega @ x + self.phase
        amp = np.sqrt(2.0 / self.weight.size)
        g = amp * float(np.cos(arg) @ self.weight)
        dg_dx = -amp * ((np.sin(arg) * self.weight) @ self.omega)
        resid = (x - self.centre) / self.rho
        value = g - 0.5 * float(resid @ resid)
        du_dx = dg_dx - resid / self.rho
        du_dcurve = _FEATURES.T @ (du_dx / self.feature_sd)
        return float(self.scale * value), self.scale * du_dcurve
